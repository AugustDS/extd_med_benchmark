import numpy as np
import os
from configparser import ConfigParser
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity
from TFGenerator import TFWrapper
from utility import get_sample_counts, get_class_names
from keras.applications.densenet import DenseNet121
import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
import h5py
import pandas as pd
import sys
import argparse
import shutil 
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time

def nn(model_dir, data_dir, results_subdir, random_seed, resolution):
    np.random.seed(random_seed)
    tf.set_random_seed(np.random.randint(1 << 31))
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    set_session(sess)

    # parser config
    config_file = model_dir+ "/config.ini"
    print("Config File Path:", config_file,flush=True)
    assert os.path.isfile(config_file)
    cp = ConfigParser()
    cp.read(config_file)

    output_dir = os.path.join(results_subdir, "classification_results/nn")
    train_outdir = os.path.join(results_subdir, "classification_results/train")
    print("Output Directory:", output_dir, flush=True)


    # default config
    image_dimension = cp["TRAIN"].getint("image_dimension")
    gan_resolution = resolution
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    if use_best_weights:
        print("** Using BEST weights",flush=True)
        model_weights_path = os.path.join(results_subdir, "classification_results/nn/best_weights.h5")
    else:
        print("** Using LAST weights",flush=True)
        model_weights_path = os.path.join(results_subdir, "classification_results/nn/weights.h5")

    print("** DenseNet Input Resolution:", image_dimension, flush=True)
    print("** GAN Image Resolution:", gan_resolution, flush=True)


    tfrecord_dir_tr = os.path.join(data_dir, "train")
    tfrecord_dir_te = os.path.join(results_subdir, "inference/test")
    # Get class names 
    class_names = get_class_names(train_outdir, "train")
    counts, _ = get_sample_counts(train_outdir, "train", class_names)
    
    # get indicies (all of csv file for validation)
    print("** counts:", counts, flush=True)
    # compute steps
    train_steps = int(np.floor(counts / batch_size))
    print("** t_steps:", train_steps, flush=True)

    log2_record = int(np.log2(gan_resolution))
    record_file_ending = "*"+ np.str(log2_record)+ ".tfrecords"
    print("** resolution ", gan_resolution, " corresponds to ", record_file_ending, " TFRecord file.", flush=True)

    # Get Model
    # ------------------------------------
    input_shape=(image_dimension, image_dimension, 3)
    img_input = Input(shape=input_shape)

    base_model = DenseNet121(
        include_top = False, 
        weights = None,
        input_tensor = img_input,
        input_shape = input_shape,
        pooling = "avg")

    x = base_model.output
    predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs = predictions)

    print(" ** load model from:", model_weights_path, flush=True)
    model.load_weights(model_weights_path)
    # ------------------------------------
    # Extract representation layer output:
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)

    #intermediate_output = intermediate_layer_model(data)

    def renorm_and_save_npy(x,name):
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        x = x*imagenet_std + imagenet_mean
        save_path = output_dir+"/"+name+".npy"
        np.save(save_path, x)
        print("** save npy images under: ", save_path, flush=True)

    def save_array(x,name):
        save_path = output_dir+"/"+name+".npy"
        np.save(save_path, x)
        print("** save npy images under: ", save_path, flush=True)

    # Load test Inference images
    test_bs = 200
    print("** load inference images, save random n=", test_bs, flush=True)
    test_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_te,
            record_file_endings = record_file_ending,
            batch_size = test_bs,
            model_target_size = (image_dimension, image_dimension),
            steps = None,
            augment=False,
            shuffle=False,
            prefetch=True,
            repeat=False)
    test_seq.initialise()
    x, x_orig, x_label = test_seq.__getitem__(0)
    renorm_and_save_npy(x,name="real_inf_224")
    renorm_and_save_npy(x_orig, name="real_inf_256")
    save_array(x_label, name="real_inf_label")

    print("** Compute inf latent rep **", flush=True)
    x_latrep = intermediate_layer_model.predict(x)
    print("** Latent Size: ", x_latrep.shape, flush=True)

    # Load train Inference images 
    print("** load train generator **", flush=True)
    train_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_tr,
            record_file_endings = record_file_ending,
            batch_size = batch_size,
            model_target_size = (image_dimension, image_dimension),
            steps = train_steps,
            augment=False,
            shuffle=False,
            prefetch=True,
            repeat=False)
    train_seq.initialise()
    print("** generator loaded **", flush =True)
    # Loop through training data and compute minimums 
    H,H_orig = image_dimension, 256
    W,W_orig = image_dimension, 256
    D = 3
    BS = batch_size
    n = test_bs
    LS = x_latrep.shape[1]
    cur_nn_imgs = np.zeros((n,H,W,D))  #Current nn images
    cur_nn_imgs_orig = np.zeros((n,H_orig,W_orig,D))
    cur_nn_labels = np.zeros((n,x_label.shape[1]))
    cur_cos_min = np.ones((n,1))*10000 #Current minimum cosine distance

    time_old = time.time()
    print("** Start nn determination **", flush =True)
    for i in range(0,train_steps):
        # Get batch images and lat. reps 
        y, y_orig, y_label = train_seq.__getitem__(i)            #[BS,H,W,D]
        y_latrep = intermediate_layer_model.predict(y) #[BS,LS]  
        
        #y_reshaped = y.reshape([BS,1,H,W,D])   #Reshape for tiling [BS,1,H,W,D]
        #y_orig_reshaped = y_orig.reshape([BS,1,H_orig,W_orig,D])
        #y_label_reshaped = y_label.reshape([BS,1,x_label.shape[1]])

        y_tiled = np.tile(y,[1,n,1,1,1])       #Tile: [BS,n,H,W,D]
        y_orig_tiled = np.tile(y_orig,[1,n,1,1,1])
        y_label_tiled = np.tile(y_label,[1,n,1])

        cosdis  = np.ones((n,BS)) - cosine_similarity(x_latrep, y_latrep) #[n,BS]
        argmin_cosdis = np.argmin(cosdis,axis=1)                          #[n,1]
        min_cosdis = np.min(cosdis,axis=1).reshape(n,1)                   #[n,1]
        
        min_y = y_tiled[:,argmin_cosdis].reshape(n,H,W,D)                 #[n,H,W,D]: Min. Cosdis for each inf_img from batch
        min_y_orig = y_orig_tiled[:,argmin_cosdis].reshape(n,H_orig,W_orig,D)
        min_ylabel = y_label_tiled[:,argmin_cosdis].reshape((n,x_label.shape[1]))

        t = np.where(min_cosdis<cur_cos_min)                              #Indicies where min. cosdistance is smaller then current 
        
        cur_cos_min[t[0]] = min_cosdis[t[0]]                              #Update current cosdis minima
        cur_nn_imgs[t[0]] = min_y[t[0]]                                    #Update current nn images 
        cur_nn_imgs_orig[t[0]] = min_y_orig[t[0]]
        cur_nn_labels[t[0]] = min_ylabel[t[0]]

        if i%100 == 0 and i>0:
            time_new = time.time()
            print("Iteration ",i, "/",train_steps, "took %.2f seconds" % (time_new - time_old))
            time_old = time_new
            print("Current mean cos-distance:", np.mean(cur_cos_min))

    print("** Loop Done **",flush=True)
    renorm_and_save_npy(cur_nn_imgs,name="nn_images_224")
    renorm_and_save_npy(cur_nn_imgs_orig,name="nn_images_256")
    save_array(cur_cos_min, name="cosdistance_minimum")
    save_array(cur_nn_labels, name="nn_labels")


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('nn',                                  'Compute NNs.')
    p.add_argument('model_dir',                    help='Model Directory')
    p.add_argument('data_dir',                 help='Data Base Directory')
    p.add_argument('results_subdir',             help='Results Directory')
    p.add_argument('random_seed',            type=int, help='Random Seed')
    p.add_argument('resolution',             type=int,  help='Resolution')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#---------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
