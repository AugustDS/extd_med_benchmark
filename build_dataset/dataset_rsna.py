# Copyright 2018, Tero Karras, NVIDIA CORPORATION
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Adapted from the original implementation by Tero Karras.
# Source https://github.com/tkarras/progressive_growing_of_gans

import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import logging
import tfutil
import dataset

#----------------------------------------------------------------------------

def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=5000):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            #assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = Queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = Queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)
           
        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1
    
        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

import pandas as pd
from PIL import Image
from skimage.transform import resize
import random 
from sklearn.model_selection import StratifiedShuffleSplit
import os

def transform_coordinates(x,w,h):
    ### Input  x: [x_top_left, y_bottom_right, w, h], w: img_width, h:img_height
    ### Output y: [x_mid, y_mid, w, h]_scaled (YOLOv5 format)
    y = np.copy(x)
    y[0] = (x[0]+x[2]/2)/w
    y[1] = (x[1]+x[3]/2)/h
    y[2] = x[2]/w
    y[3] = x[3]/h
    return y 

def return_gan_label(y):
    if y == 0:
        return([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    else:
        return([0,0,0,0,1,0,0,0,1,0,0,0,0,0,0])

def create_from_rsna(save_dir, image_dir, limit = 256, split = 0.1, np_seed=100, res=128):

    path_dt = image_dir #/work/aschuette/pneumonia-challenge-dataset-adjudicated-kaggle_2018
    print("----------------------------------------------", flush=True)
    print('Loading images from "%s"' % path_dt, flush=True)
    data_frame = pd.read_csv(path_dt + "/stage_2_train_labels.csv")
    header = ["Path","No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema",
        "Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
    
    tfrecord_dir = save_dir + "/tf_rsna_dataset"
    objdetec_dir = save_dir + "/od_rsna_dataset"

    unique_ids = np.unique(data_frame["patientId"])
    rd_index = np.arange(0,unique_ids.shape[0],1)
    np.random.RandomState(np_seed).shuffle(rd_index)
    train_id = unique_ids[rd_index[0:int((1-2*split)*len(rd_index))]]
    valid_id = unique_ids[rd_index[int((1-2*split)*len(rd_index)):int((1-split)*len(rd_index))]]
    test_id  = unique_ids[rd_index[int((1-split)*len(rd_index)):]]
    x_tr=[];  y_tr=[]; y_od_tr = []
    x_val=[]; y_val=[]; y_od_val = []
    x_te=[];  y_te=[]; y_od_te = []

    for row in data_frame.iterrows():
        if row[1][0] in train_id:
            x_tr.append(row[1][0])
            y_tr.append(return_gan_label(row[1][-1]))
            y_od_tr.append([np.float(row[1][-1]),np.float(row[1][1]),np.float(row[1][2]),np.float(row[1][3]),np.float(row[1][4])])
        elif row[1][0] in valid_id:
            x_val.append(row[1][0])
            y_val.append(return_gan_label(row[1][-1]))
            y_od_val.append([np.float(row[1][-1]),np.float(row[1][1]),np.float(row[1][2]),np.float(row[1][3]),np.float(row[1][4])])
        elif row[1][0] in test_id:
            x_te.append(row[1][0])
            y_te.append(return_gan_label(row[1][-1]))
            y_od_te.append([np.float(row[1][-1]),np.float(row[1][1]),np.float(row[1][2]),np.float(row[1][3]),np.float(row[1][4])])

    x_tr=np.asarray(x_tr).reshape(-1,1);y_tr=np.asarray(y_tr); y_od_tr=np.asarray(y_od_tr)
    x_val=np.asarray(x_val).reshape(-1,1);y_val=np.asarray(y_val); y_od_val=np.asarray(y_od_val)
    x_te=np.asarray(x_te).reshape(-1,1);y_te=np.asarray(y_te); y_od_te=np.asarray(y_od_te)
    
    tfrecord_dir_tr = tfrecord_dir+"/train"
    tfrecord_dir_vl = tfrecord_dir+"/valid"
    tfrecord_dir_te = tfrecord_dir+"/test"
    if not os.path.isdir(tfrecord_dir_tr):
        os.makedirs(tfrecord_dir_tr)
    if not os.path.isdir(tfrecord_dir_vl):
        os.makedirs(tfrecord_dir_vl)
    if not os.path.isdir(tfrecord_dir_te):
        os.makedirs(tfrecord_dir_te)

    tr_img_path = os.path.join(objdetec_dir, "images/train")
    tr_txt_path = os.path.join(objdetec_dir, "labels/train")
    vl_img_path = os.path.join(objdetec_dir, "images/val")
    vl_txt_path = os.path.join(objdetec_dir, "labels/val")
    te_img_path = os.path.join(objdetec_dir, "images/test")
    te_txt_path = os.path.join(objdetec_dir, "labels/test")

    all_paths = [tr_img_path,tr_txt_path,vl_img_path,vl_txt_path,te_img_path,te_txt_path]
    for check_p in all_paths:
        if not os.path.isdir(check_p):
            os.makedirs(check_p)

    print("Save train.csv file to:", tfrecord_dir_tr+"/train.csv", flush=True)
    train_data = np.concatenate((x_tr,y_tr),axis=1)
    df_tr = pd.DataFrame(columns=header,data=train_data)
    df_tr.to_csv(tfrecord_dir_tr+"/train.csv", mode='w', header=True,index=False)

    print("Save validation.csv file to:", tfrecord_dir_vl+"/valid.csv", flush=True)
    val_data = np.concatenate((x_val,y_val),axis=1)
    df_vl = pd.DataFrame(columns=header,data=val_data)
    df_vl.to_csv(tfrecord_dir_vl+"/valid.csv", mode='w', header=True,index=False)

    print("Save test.csv file to:", tfrecord_dir_te+"/test.csv", flush=True)
    test_data = np.concatenate((x_te,y_te),axis=1)
    df_te = pd.DataFrame(columns=header,data=test_data)
    df_te.to_csv(tfrecord_dir_te+"/test.csv", mode='w', header=True,index=False)

    print('Train data shape (path+labels):', train_data.shape, flush=True)
    print('Validation data shape (path+labels):', val_data.shape, flush=True)
    print('Test data shape (path+labels):', test_data.shape, flush=True)
    print("----------------------------------------------", flush=True)

    print("BUILD TFRECORD FILES", flush=True)

    from pydicom import dcmread
    from PIL import Image
    # Train TFR
    # --------------------------------------------
    images    = []
    for i in range(len(x_tr)):
        dicom_file = dcmread(path_dt+"/stage_2_train_images/"+x_tr[i][0]+".dcm")
        image_2d = dicom_file.pixel_array.astype(float)
        image_2d = resize(image_array, (res,res))
        image_2d = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        image_2d = np.uint8(image_2d)

        ### YOLO
        cl,bbx_orig = y_od_tr[i][0],y_od_tr[i][1:]
        path_txt = os.path.join(tr_txt_path, x_tr[i][0]+".txt")
        path_img = os.path.join(tr_img_path, x_tr[i][0]+".jpg")
        bbx = transform_coordinates(bbx_orig,w=res,h=res)
        line_txt = str(cl)+" "+str(bbx[0])+" "+str(bbx[1])+" "+str(bbx[2])+" "+str(bbx[3])+"\n"
        f = open(path_txt,"a")
        if cl == 1.:
            f.write(line_txt)
        f.close()
        image_save  = Image.fromarray(image_2d)
        image_save.save(path_img)

        ### GAN TF
        image_array = image_2d.reshape(1,res,res)
        images.append(image_array)

    images = np.asarray(images)
    print('Train images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_tr, images.shape[0], progress_interval=5000) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(y_tr[order])
    
    print('Train data done.', flush=True)



    images    = []
    for i in range(len(x_val)):
        dicom_file = dcmread(path_dt+"/stage_2_train_images/"+x_val[i][0]+".dcm")
        image_2d = dicom_file.pixel_array.astype(float)
        image_2d = resize(image_array, (res,res))
        image_2d = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        image_2d = np.uint8(image_2d)

        ### YOLO
        cl,bbx_orig = y_od_val[i][0],y_od_val[i][1:]
        path_txt = os.path.join(vl_txt_path, x_val[i][0]+".txt")
        path_img = os.path.join(vl_img_path, x_val[i][0]+".jpg")
        bbx = transform_coordinates(bbx_orig,w=res,h=res)
        line_txt = str(cl)+" "+str(bbx[0])+" "+str(bbx[1])+" "+str(bbx[2])+" "+str(bbx[3])+"\n"
        f = open(path_txt,"a")
        if cl == 1.:
            f.write(line_txt)
        f.close()
        image_save  = Image.fromarray(image_2d)
        image_save.save(path_img)

        ### GAN TF
        image_array = image_2d.reshape(1,res,res)
        images.append(image_array)

    images = np.asarray(images)
    print('Train images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_vl, images.shape[0], progress_interval=5000) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(y_val[order])
    
    print('Val data done.', flush=True)

    images    = []
    for i in range(len(x_te)):
        dicom_file = dcmread(path_dt+"/stage_2_train_images/"+x_te[i][0]+".dcm")
        image_2d = dicom_file.pixel_array.astype(float)
        image_2d = resize(image_array, (res,res))
        image_2d = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
        image_2d = np.uint8(image_2d)

        ### YOLO
        cl,bbx_orig = y_od_te[i][0],y_od_te[i][1:]
        path_txt = os.path.join(te_txt_path, x_te[i][0]+".txt")
        path_img = os.path.join(te_img_path, x_te[i][0]+".jpg")
        bbx = transform_coordinates(bbx_orig,w=res,h=res)
        line_txt = str(cl)+" "+str(bbx[0])+" "+str(bbx[1])+" "+str(bbx[2])+" "+str(bbx[3])+"\n"
        f = open(path_txt,"a")
        if cl == 1.:
            f.write(line_txt)
        f.close()
        image_save  = Image.fromarray(image_2d)
        image_save.save(path_img)

        ### GAN TF
        image_array = image_2d.reshape(1,res,res)
        images.append(image_array)

    images = np.asarray(images)
    print('Train images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_te, images.shape[0], progress_interval=5000) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(y_te[order])
    
    print('Test data done.', flush=True)

    print("----------------------------------------------", flush=True)
    print("DONE", flush=True)
    print("----------------------------------------------", flush=True)


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating, extracting, and visualizing Progressive GAN datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command( 'create_from_rsna', 'Create dataset from xray iamges.')
    p.add_argument(     'save_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',    help='Directory containing the images')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
