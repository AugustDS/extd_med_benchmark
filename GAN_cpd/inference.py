import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import pandas as pd
import os 
import sys
import argparse
import PIL

import os
import glob
import numpy as np
import tensorflow as tf
import tfutil

#----------------------------------------------------------------------------
# Parse individual image from a tfrecords file.

def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.

class TFRecordDataset:
    def __init__(self,
        tfrecord_file,              # Tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 2):       # Number of concurrent threads.

        self.tfrecord_file      = tfrecord_file
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channel, height, width]
        self.dtype              = 'uint8'
        self.dynamic_range      = [0, 255]
        self.label_file         = label_file
        self.label_size         = None      # [component]
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1

        assert os.path.isfile(self.tfrecord_file)
        assert os.path.isfile(self.label_file)
        tfr_shapes = []
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(self.tfrecord_file, tfr_opt):
            tfr_shapes.append(parse_tfrecord_np(record).shape)
            break

        tfr_files = [self.tfrecord_file]

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
        self.resolution = resolution if resolution is not None else max_shape[1]
        print("----------------------------", flush=True)
        print("Resolution used:", self.resolution, flush=True)
        print("----------------------------", flush=True)
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))

        print("tfr_lods:", tfr_lods, flush=True)
        print("tfr_shapes:", tfr_shapes, flush=True)
        print("tfr_files:", tfr_files, flush=True)

        #assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<20, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            tf_labels_init = tf.zeros(self._np_labels.shape, self._np_labels.dtype)
            self._tf_labels_var = tf.Variable(tf_labels_init, name='labels_var')
            tfutil.set_vars({self._tf_labels_var: self._np_labels})
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
            for tfr_file, tfr_shape, tfr_lod in zip(tfr_files, tfr_shapes, tfr_lods):
                if tfr_lod < 0:
                    continue
                dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=buffer_mb<<20)
                dset = dset.map(parse_tfrecord_tf, num_parallel_calls=num_threads)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
                bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
                dset = dset.batch(self._tf_minibatch_in)
                self._tf_datasets[tfr_lod] = dset
            self._tf_iterator = tf.data.Iterator.from_structure(self._tf_datasets[0].output_types, self._tf_datasets[0].output_shapes)
            self._tf_init_ops = {lod: self._tf_iterator.make_initializer(dset) for lod, dset in self._tf_datasets.items()}

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 and lod in self._tf_datasets
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            self._tf_init_ops[lod].run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        if self._tf_minibatch_np is None:
            self._tf_minibatch_np = self.get_minibatch_tf()
        return tfutil.run(self._tf_minibatch_np)

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        if self.label_size > 0:
            return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
        else:
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        else:
            return np.zeros([minibatch_size, 0], self.label_dtype)


def inference(data_dir, result_subdir, random_seed, batch_size = 20): 

    model_name_path = result_subdir + "/network-final-full-conv.pkl"
    print("Loading model: ", model_name_path, flush=True)
    print("Data Base Dir:", data_dir, flush=True)

    tf_test_record_file = data_dir + "/test/test-r08.tfrecords"
    tf_test_label_file  = data_dir + "/test/test-rxx.labels"

    tf.InteractiveSession()

    dataset = TFRecordDataset(tfrecord_file=tf_test_record_file,
                        label_file=tf_test_label_file, resolution=256, max_label_size = "full")

    csv_input_test  = data_dir + "/test/test.csv"
    csv_input_train = data_dir + "/train/train.csv"
    csv_input_valid = data_dir + "/valid/valid.csv"
    inf_path_test = result_subdir + "/inference/test_pngs"

    inf_path_reals = inf_path_test + "/reals"
    inf_path_fakes = inf_path_test + "/fakes"
    csv_save_fakes = inf_path_fakes + "/test.csv"
    csv_save_reals = inf_path_reals + "/test.csv"

    if not os.path.exists(inf_path_test):
        os.makedirs(inf_path_test)

    if not os.path.exists(inf_path_reals):
        os.makedirs(inf_path_reals)

    if not os.path.exists(inf_path_fakes):
        os.makedirs(inf_path_fakes)

    data_frame_te = pd.read_csv(csv_input_test)
    data_frame_tr = pd.read_csv(csv_input_train)
    data_frame_vl = pd.read_csv(csv_input_valid)

    np_labels_tr = []
    np_labels_vl = []

    for row in data_frame_tr.iterrows():
        np_labels_tr.append(row[1][1:].values)

    for row in data_frame_vl.iterrows():
        np_labels_vl.append(row[1][1:].values)

    labels_arr_tr   = np.asarray(np_labels_tr)
    num_examples_tr = labels_arr_tr.shape[0]
    num_batches_tr  = np.int(np.ceil(num_examples_tr/batch_size))
    labels_arr_vl   = np.asarray(np_labels_vl)
    num_examples_vl = labels_arr_vl.shape[0]
    num_batches_vl  = np.int(np.ceil(num_examples_vl/batch_size))

    split_headers = list(data_frame_te.columns[0:]) #Remove "Path" from names

    paths_te = []
    np_labels_te = []

    for row in data_frame_te.iterrows():
        np_labels_te.append(row[1][1:].values)

    labels_arr_te   = np.asarray(np_labels_te)
    num_examples_te = labels_arr_te.shape[0]
    num_batches_te  = np.int(np.ceil(num_examples_te/batch_size))

    print("Test Label Data Shape:", labels_arr_te.shape,flush=True)
    print("Number Test Batches:", num_batches_te,flush=True)

    # Import official network.
    with open(model_name_path, 'rb') as file:
        all_models = pickle.load(file)
        Gs = all_models[-1]

    # Create CSV File Test Fake
    ids_i_te   = np.arange(0,labels_arr_te.shape[0],1)
    ids_te_fake = np.array([inf_path_fakes+"/"+np.str(np.int(ids_i_te[j])) + ".png" for j in range(0,len(ids_i_te))]).reshape(labels_arr_te.shape[0],1)
    ids_and_labs_te = np.concatenate((ids_te_fake,labels_arr_te),axis=1)
    df_new_te = pd.DataFrame(columns=split_headers,data=ids_and_labs_te)
    df_new_te.to_csv(csv_save_fakes, mode='w', header=True,index=False)

    ids_te_real = np.array([inf_path_reals+"/"+np.str(np.int(ids_i_te[j])) + ".png" for j in range(0,len(ids_i_te))]).reshape(labels_arr_te.shape[0],1)

    # Generate Latents
    latents_all = np.random.RandomState(random_seed).randn(num_examples_tr+num_examples_vl+num_examples_te,*Gs.input_shapes[0][1:])
    # Split latents for train, val, test
    latents_te  = latents_all[num_examples_tr+num_examples_vl:,:]
    assert latents_te.shape[0] == num_examples_te

    def adjust_dynamic_range(data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def convert_to_pil_image(image, drange=[0,1]):
        assert image.ndim == 2 or image.ndim == 3
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0] # grayscale CHW => HW
            else:
                image = image.transpose(1, 2, 0) # CHW -> HWC

        #image = adjust_dynamic_range(image, drange, [0,255])
        #image = np.rint(image).clip(0, 255).astype(np.uint8)
        format = 'RGB' if image.ndim == 3 else 'L'
        return PIL.Image.fromarray(image, format)


    def save_image(image, filename, drange=[0,1], quality=95):
        img = convert_to_pil_image(image, drange)
        img.save(filename)


    print('Generating inference test images..', flush=True)
    img_i = 0
    all_labels_rl = []
    for i in range(0,num_batches_te):
        labels  = labels_arr_te[i*batch_size:(i+1)*batch_size]
        latents = latents_te[i*batch_size:(i+1)*batch_size]
        images_fake  = Gs.run(latents, labels)
        images_real, labels_real = dataset.get_minibatch_np(minibatch_size=batch_size)
        images_fake = np.clip(np.rint((images_fake + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
        for j in range(0,len(images_fake)):
            img_fk = images_fake[j]
            img_rl = images_real[j]
            name_fk = ids_te_fake[img_i][0]
            name_rl = ids_te_real[img_i][0]
            save_image(img_fk, name_fk)
            save_image(img_rl, name_rl)
            all_labels_rl.append(labels_real[j])
            img_i += 1

    all_labels_rl = np.asarray(all_labels_rl)
    # Create CSV File Test Real
    ids_and_labs_te_rl = np.concatenate((ids_te_real,all_labels_rl),axis=1)
    df_new_te = pd.DataFrame(columns=split_headers,data=ids_and_labs_te_rl)
    df_new_te.to_csv(csv_save_reals, mode='w', header=True,index=False)

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('inference',                               'Inference.')
    p.add_argument('data_dir',                        help='Data load Path')
    p.add_argument('result_subdir',                help='Results Directory')
    p.add_argument('random_seed',              type=int, help='Random Seed')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
