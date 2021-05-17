import numpy as np
import os
from configparser import ConfigParser
from sklearn.metrics import roc_auc_score, roc_curve
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
from tensorflow.python.keras.losses import categorical_crossentropy
from cxplain import UNetModelBuilder, ZeroMasking, CXPlain
from sklearn import model_selection
from cxplain.backend.serialisation.tf_model_serialisation import TensorFlowModelSerialiser
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def cxpl(model_dir, data_dir, results_subdir, random_seed, resolution):
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

    output_dir = os.path.join(results_subdir, "classification_results/test")
    print("Output Directory:", output_dir,flush=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    # default config
    image_dimension = cp["TRAIN"].getint("image_dimension")
    gan_resolution = resolution
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    if use_best_weights:
        print("** Using BEST weights",flush=True)
        model_weights_path = os.path.join(results_subdir, "classification_results/train/best_weights.h5")
    else:
        print("** Using LAST weights",flush=True)
        model_weights_path = os.path.join(results_subdir, "classification_results/train/weights.h5")

    print("** DenseNet Input Resolution:", image_dimension, flush=True)
    print("** GAN Image Resolution:", gan_resolution, flush=True)

    # get test sample count
    test_dir = os.path.join(results_subdir, "inference/test")
    shutil.copy(test_dir+"/test.csv", output_dir)

    # Get class names 
    class_names = get_class_names(output_dir,"test")

    tfrecord_dir_te = os.path.join(data_dir, "test")
    test_counts, _ = get_sample_counts(output_dir, "test", class_names)
    
    # get indicies (all of csv file for validation)
    print("** test counts:", test_counts, flush=True)

    # compute steps
    test_steps = int(np.floor(test_counts / batch_size))
    print("** test_steps:", test_steps, flush=True)

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

    print("** load test generator **", flush=True)
    test_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_te,
            record_file_endings = record_file_ending,
            batch_size = batch_size,
            model_target_size = (image_dimension, image_dimension),
            steps = None,
            augment=False,
            shuffle=False,
            prefetch=True,
            repeat=False)

    print("** make prediction **", flush=True)
    test_seq.initialise() 
    x_all, y_all = test_seq.get_all_test_data()
    print("X-Test  Shape:", x_all.shape,flush=True)
    print("Y-Test  Shape:", y_all.shape,flush=True)

    print("----------------------------------------", flush=True)
    print("Test Model AUROC", flush=True)
    y_pred = model.predict(x_all)
    current_auroc = []
    for i in range(len(class_names)):
        try:
            score = roc_auc_score(y_all[:, i], y_pred[:, i])
        except ValueError:
            score = 0
        current_auroc.append(score)
        print(i+1,class_names[i],": ", score, flush=True)
    mean_auroc = np.mean(current_auroc)
    print("Mean auroc: ", mean_auroc,flush=True)

    print("----------------------------------------", flush=True)
    downscale_factor  = 8
    num_models_to_use = 3
    num_test_images   = 100
    print("Number of Models to use:", num_models_to_use, flush=True)
    print("Number of Test images:", num_test_images, flush=True)
    x_tr, y_tr = x_all[num_test_images:], y_all[num_test_images:]
    x_te, y_te = x_all[0:num_test_images], y_all[0:num_test_images]

    downsample_factors = (downscale_factor,downscale_factor)
    print("Downsample Factors:", downsample_factors,flush=True)
    model_builder = UNetModelBuilder(downsample_factors, num_layers=2, num_units=8, activation="relu",
                                     p_dropout=0.0, verbose=0, batch_size=32, learning_rate=0.001)
    print("Model build done.",flush=True)
    masking_operation = ZeroMasking()
    loss = categorical_crossentropy

    explainer = CXPlain(model, model_builder, masking_operation, loss, 
                    num_models=num_models_to_use, downsample_factors=downsample_factors, flatten_for_explained_model=False)
    print("Explainer build done.",flush=True)

    explainer.fit(x_tr, y_tr);
    print("Explainer fit done.",flush=True)

    try:
        attr, conf = explainer.explain(x_te, confidence_level=0.80)
        np.save(output_dir+"/x_cxpl.npy", x_te)
        np.save(output_dir+"/y_cxpl.npy", y_te)
        np.save(output_dir+"/attr.npy", attr)
        np.save(output_dir+"/conf.npy", conf)
        print("Explainer explain done and saved.",flush=True)
    except Exception as ef: print(ef,flush=True)

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('cxpl',                            'Test Classifier.')
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
