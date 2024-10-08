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

def create_from_xray(tfrecord_dir, image_dir, limit = 256, split = 0.1, test_val_samples=4000,np_seed=100):

    path_dt = image_dir + "/CheXpert-v1.0"
    print("----------------------------------------------", flush=True)
    print('Loading images from "%s"' % path_dt, flush=True)
    data_frame = pd.read_csv(path_dt + "/train.csv")
    filename_1 = "/view1_frontal.jpg"
    filename_2 = "/view2_frontal.jpg"
    header = ["Path","No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema",
        "Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
    
    paths = []
    labels = []
    ids = []

    for row in data_frame.iterrows():
        if (filename_1 in row[1][0]) or (filename_2 in row[1][0]):
            x = row[1][5:].values
            x[x == -1.0] = 1.0          #Turn all uncertain labels positive
            x[(x!=0.0)&(x!=1.0)] = 0.0  #Turn all empties into 0
            if np.sum(x)>0:
                labels.append(x.astype(np.float32))
                paths.append(image_dir + "/" + row[1][0])
                ids.append(row[1][0].split("patient")[1].split("/")[0])

    all_labels = np.asarray(labels)
    all_paths = np.asarray(paths)
    patient_id = np.asarray(ids)

    # Get classes with freq>limit
    dt = np.dtype((np.void, all_labels.dtype.itemsize * all_labels.shape[1]))
    b = np.ascontiguousarray(all_labels).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(all_labels.dtype).reshape(-1, all_labels.shape[1])
    print("Number of unique classes: ", len(cnt))
    idx = np.argsort(cnt)
    sorted_index = idx[::-1]
    sorted_count = cnt[sorted_index]
    lim = np.where(sorted_count>limit)[0][-1]+1
    print("Number of remaining Classes:", lim, "out of", len(cnt), flush=True)
    percentile=np.sum(sorted_count[0:lim])/np.sum(sorted_count)
    print("They make up %.1f Percent of Training Data."%(percentile*100), flush=True)
    classes_remain = unq[sorted_index[0:lim]]
    y_new = []
    x_new = []
    id_new = []
    for i in range(0,len(all_labels)):
        for j in range(0,len(classes_remain)):
            if (all_labels[i,:]==classes_remain[j,:]).all():
                y_new.append(all_labels[i,:])
                x_new.append(all_paths[i])
                id_new.append(patient_id[i])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)
    id_new = np.asarray(id_new)
    print("----------------------------------------------", flush=True)
    
    print("Split into train, valid, test on patient level:",1-split,"/", split, "/", split, flush=True)
    # Split train, test, valid on patient level
    unq_id = np.unique(id_new)
    rd_index = np.arange(0,unq_id.shape[0],1)
    np.random.RandomState(np_seed).shuffle(rd_index)
    np_seed += 1
    train_id = unq_id[rd_index[0:int((1-2*split)*len(rd_index))]]
    valid_id = unq_id[rd_index[int((1-2*split)*len(rd_index)):int((1-split)*len(rd_index))]]
    test_id  = unq_id[rd_index[int((1-split)*len(rd_index)):]]

    x_tr=[];y_tr=[]; id_tr=[]
    x_val=[];y_val=[]; id_val=[]
    x_te=[]; y_te=[];  id_te=[]

    for i in range(0,y_new.shape[0]):
        if id_new[i] in train_id:
            x_tr.append(x_new[i])
            y_tr.append(y_new[i])
            id_tr.append(id_new[i])
        elif id_new[i] in valid_id:
            x_val.append(x_new[i])
            y_val.append(y_new[i])
            id_val.append(id_new[i])
        elif id_new[i] in test_id:
            x_te.append(x_new[i])
            y_te.append(y_new[i])
            id_te.append(id_new[i])
        else:
            raise ValueError("ID mismatch")
    
    X_train=np.asarray(x_tr);y_train=np.asarray(y_tr); id_tr=np.asarray(id_tr)
    x_val=np.asarray(x_val);y_val=np.asarray(y_val); id_val=np.asarray(id_val)
    x_te=np.asarray(x_te);y_te=np.asarray(y_te); id_te=np.asarray(id_te)
    
    print("Train patients",len(train_id), " with number of samples:", len(x_tr), flush=True)
    print("Valid patients",len(valid_id), " with number of samples:", len(x_val), flush=True)
    print("Test patients",len(test_id), " with number of samples:", len(x_te), flush=True)
    
    print("----------------------------------------------", flush=True)
    print("Extract number of validation and test images:", test_val_samples, flush=True)

    # Extract correct number of validation and test images 
    val_size = 1-test_val_samples/x_val.shape[0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=np_seed)
    sss.get_n_splits(x_val, y_val)
    for index,_ in sss.split(x_val, y_val):
        X_valid = x_val[index]
        y_valid = y_val[index]

    np_seed += 1
    test_size = 1-test_val_samples/x_te.shape[0] - 0.00005
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=np_seed)
    sss.get_n_splits(x_te, y_te)
    for index,_ in sss.split(x_te, y_te):
        X_test = x_te[index]
        y_test = y_te[index]
    
    # Validation Data
    val_labels = y_valid
    val_paths  = X_valid.reshape(X_valid.shape[0],1)
    # Test Data
    test_labels = y_test
    test_paths  = X_test.reshape(X_test.shape[0],1)
    # Train Data
    train_labels = y_train
    train_paths  = X_train.reshape(X_train.shape[0],1)
    

    print("----------------------------------------------", flush=True)
    print("CREATE DIR and SAVE CSV DATA", flush=True)
    if not os.path.isdir(tfrecord_dir):
        print("Creating directory:", tfrecord_dir, flush=True)
        os.makedirs(tfrecord_dir)
    assert(os.path.isdir(tfrecord_dir))

    tfrecord_dir_tr = tfrecord_dir+"/train"
    tfrecord_dir_vl = tfrecord_dir+"/valid"
    tfrecord_dir_te = tfrecord_dir+"/test"

    if not os.path.isdir(tfrecord_dir_tr):
        print("Creating directory:", tfrecord_dir_tr, flush=True)
        os.makedirs(tfrecord_dir_tr)
    assert(os.path.isdir(tfrecord_dir_tr))

    if not os.path.isdir(tfrecord_dir_vl):
        print("Creating directory:", tfrecord_dir_vl, flush=True)
        os.makedirs(tfrecord_dir_vl)
    assert(os.path.isdir(tfrecord_dir_vl))

    if not os.path.isdir(tfrecord_dir_te):
        print("Creating directory:", tfrecord_dir_te, flush=True)
        os.makedirs(tfrecord_dir_te)
    assert(os.path.isdir(tfrecord_dir_te))

    # Save csv files
    print("Save train.csv file to:", tfrecord_dir_tr+"/train.csv", flush=True)
    train_data = np.concatenate((train_paths,train_labels),axis=1)
    df_tr = pd.DataFrame(columns=header,data=train_data)
    df_tr.to_csv(tfrecord_dir_tr+"/train.csv", mode='w', header=True,index=False)

    print("Save validation.csv file to:", tfrecord_dir_vl+"/valid.csv", flush=True)
    val_data = np.concatenate((val_paths,val_labels),axis=1)
    df_vl = pd.DataFrame(columns=header,data=val_data)
    df_vl.to_csv(tfrecord_dir_vl+"/valid.csv", mode='w', header=True,index=False)

    print("Save test.csv file to:", tfrecord_dir_te+"/test.csv", flush=True)
    test_data = np.concatenate((test_paths,test_labels),axis=1)
    df_te = pd.DataFrame(columns=header,data=test_data)
    df_te.to_csv(tfrecord_dir_te+"/test.csv", mode='w', header=True,index=False)

    print('Train data shape (path+labels):', train_data.shape, flush=True)
    print('Validation data shape (path+labels):', val_data.shape, flush=True)
    print('Test data shape (path+labels):', test_data.shape, flush=True)
    print("----------------------------------------------", flush=True)

    print("BUILD TFRECORD FILES", flush=True)
    # Train TFR
    # --------------------------------------------
    images    = []
    for path in train_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        image_array = resize(image_array, (1024,1024))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,1024,1024)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == train_labels.shape[0]
    print('Train images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_tr, images.shape[0], progress_interval=5000) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(train_labels[order])
    # --------------------------------------------
    # Valid TFR
    # --------------------------------------------
    images    = []
    for path in val_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        image_array = resize(image_array, (1024,1024))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,1024,1024)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == val_labels.shape[0]
    print('Valid images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_vl, images.shape[0], progress_interval=500) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(val_labels[order])
    # --------------------------------------------
    # Test TFR
    # --------------------------------------------
    images    = []
    for path in test_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        image_array = resize(image_array, (1024,1024))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,1024,1024)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == test_labels.shape[0]
    print('Test images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_te, images.shape[0], progress_interval=500) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(test_labels[order])

    print("----------------------------------------------", flush=True)
    print("DONE", flush=True)
    print("----------------------------------------------", flush=True)

### Samples per class
def get_above_freq(x,y,z,classes_remain):
    y_new = []
    x_new = []
    z_new = []
    cnt = np.zeros((classes_remain.shape[0],))
    for i in range(0,len(y)):
        for j in range(0,len(classes_remain)):
            if (y[i,:]==classes_remain[j,:]).all():
                y_new.append(y[i,:])
                x_new.append(x[i])
                cnt[j] +=1
                if z is not None:
                    z_new.append(z[i])
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)
    z_new = np.asarray(z_new)
    print("Samples per class:", cnt,flush=True)
    print("Percentile kept is %.2f" %(x_new.shape[0]/x.shape[0]*100),flush=True)
    if z is not None:
        return x_new, y_new, z_new
    else:
        return x_new, y_new

def get_class_groups(x,y,classes_remain,size=None):
    seed = 0
    y_group = []
    x_group = []
    for lab in np.flip(classes_remain,axis=0):
        y_ = []
        x_ = []
        for i in range(0,y.shape[0]):
            if (y[i,:]==lab).all():
                y_.append(y[i,:])
                x_.append(x[i])
        # Shuffle and append
        y_ = np.asarray(y_)
        x_ = np.asarray(x_)
        rd_index = np.arange(0,y_.shape[0],1)
        np.random.RandomState(seed).shuffle(rd_index)
        if size is not None:
            y_group.append(y_[rd_index][0:size])
            x_group.append(x_[rd_index][0:size])
        else:
            y_group.append(y_[rd_index])
            x_group.append(x_[rd_index])
        seed +=1
    return x_group, y_group




def create_from_brain(tfrecord_dir, image_dir, no_finding_length = 107365, 
                      test_val_samples = 4000, m = 6, lim=110, split=0.1):

    path_dt = image_dir + "/new_stage_2_train.csv"
    print('Loading csv File from', path_dt, flush=True)
    raw_data_csv = pd.read_csv(path_dt)
    raw_data = raw_data_csv.values

    header = ["path", "epidural","intraparenchymal","intraventricular","subarachnoid","subdural","any"]
    final_data_frame = pd.DataFrame(columns=header)
    def plot_label_distr(x,N,h=header):
        for i in range(0,len(h)-1):
            print("Label",h[i+1],":",np.count_nonzero(x[:,i]!=0.0),"/",N,"=",np.count_nonzero(x[:,i]!=0.0)/N,"%")

    def plot_class_distr(x):
        dt = np.dtype((np.void, x.dtype.itemsize * x.shape[1]))
        b = np.ascontiguousarray(x).view(dt)
        unq, cnt = np.unique(b, return_counts=True)
        unq = unq.view(x.dtype).reshape(-1, x.shape[1])
        print("Number of unique classes: ", len(cnt), flush=True)
        idx = np.argsort(cnt)
        sorted_index = idx[::-1]
        sorted_count = cnt[sorted_index]
        print(sorted_count)

    print("GET DATA", flush=True)
    n = int(raw_data.shape[0]/m)

    y_label = np.ones((n,m))*100
    y_ids   = []
    pat_id     = []
    for i in range(0,n):
        list_labs = []
        ID_ = raw_data[i*m,0].split("_")[1]
        pat_id.append(raw_data[i*m,2])
        for j in range (0,m):
            ID,class_name,label = raw_data[i*m+j,0].split("_")[1], raw_data[i*m+j,0].split("_")[2], raw_data[i*m+j,1]
            assert class_name == header[j+1]
            assert ID == ID_
            if label == 0.0:
                list_labs.append(0.0)
            else:
                list_labs.append(1.0)
        y_ids.append(image_dir+"/ID_"+ID_ + ".jpg")
        y_label[i,:] = np.asarray(list_labs)

    for i in range(0,y_label.shape[0]):
        if np.sum(y_label[i,0:-1])==0.0:
            assert y_label[i,-1]==0.0
            y_label[i,-1] = 1.0
        else:
            y_label[i,-1] = 0.0
    header[-1] = "no finding"

    y_ids = np.asarray(y_ids)
    pat_id = np.asarray(pat_id)

    split_headers = header
    print("Original Distribution:", flush=True)
    plot_label_distr(y_label,n)

    print("----------------------------------------------", flush=True)
    print("Remove classes below frequency:", lim, flush=True)
    dt = np.dtype((np.void, y_label.dtype.itemsize * y_label.shape[1]))
    b = np.ascontiguousarray(y_label).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(y_label.dtype).reshape(-1, y_label.shape[1])
    print("Number of unique classes: ", len(cnt), flush=True)
    idx = np.argsort(cnt)
    sorted_index = idx[::-1]
    sorted_count = cnt[sorted_index]
    lim = np.where(sorted_count>lim)[0][-1]+1
    print("Number of remaining Classes:", lim, "out of", len(cnt), flush=True)
    percentile=np.sum(sorted_count[0:lim])/np.sum(sorted_count)
    print("They make up %.1f Percent of Training Data."%(percentile*100), flush=True)
    classes_remain = unq[sorted_index[0:lim]]
    y_new = []
    x_new = []
    p_new = []
    for i in range(0,len(y_label)):
        for j in range(0,len(classes_remain)):
            if (y_label[i,:]==classes_remain[j,:]).all():
                y_new.append(y_label[i,:])
                x_new.append(y_ids[i])
                p_new.append(pat_id[i])
    y_ids = np.asarray(x_new)
    y_label = np.asarray(y_new)
    pat_id = np.asarray(p_new)
    print("New Distribution:", flush=True)
    plot_label_distr(y_label,N=y_label.shape[0])
    print("----------------------------------------------", flush=True)


    print("BALANCE DATA", flush=True)
    np_seed = 1000
    index = np.where(y_label[:,-1]==1.0)[0]
    np.random.RandomState(np_seed).shuffle(index)
    np_seed += 1
    y_no_finding  = y_label[index,:]
    id_no_finding = y_ids[index]
    pat_id_no_finding = pat_id[index]

    y_any  = np.delete(y_label, index, axis=0)
    id_any = np.delete(y_ids, index, axis=0)
    pat_id_any = np.delete(pat_id, index, axis=0)

    y_new = np.concatenate((y_any, y_no_finding[0:no_finding_length,:]),axis=0)
    x_new = np.concatenate((id_any, id_no_finding[0:no_finding_length]), axis=0)
    pat_id_new = np.concatenate((pat_id_any, pat_id_no_finding[0:no_finding_length]), axis=0)
    n_balanced = len(y_new)
    plot_label_distr(y_new,n_balanced)

    print("----------------------------------------------", flush=True)
    print("Split into train, valid, test on patient level:",1-split,"/", split, "/", split, flush=True)

    # Split train, test, valid on patient level
    unq_id = np.unique(pat_id_new)
    rd_index = np.arange(0,unq_id.shape[0],1)
    np.random.RandomState(np_seed).shuffle(rd_index)
    np_seed += 1
    train_id = unq_id[rd_index[0:int((1-2*split)*len(rd_index))]]
    valid_id = unq_id[rd_index[int((1-2*split)*len(rd_index)):int((1-split)*len(rd_index))]]
    test_id  = unq_id[rd_index[int((1-split)*len(rd_index)):]]

    x_tr=[];y_tr=[]; id_tr=[]
    x_val=[];y_val=[]; id_val=[]
    x_te=[]; y_te=[];  id_te=[]

    for i in range(0,y_new.shape[0]):
        if pat_id_new[i] in train_id:
            x_tr.append(x_new[i])
            y_tr.append(y_new[i])
            id_tr.append(pat_id_new[i])
        elif pat_id_new[i] in valid_id:
            x_val.append(x_new[i])
            y_val.append(y_new[i])
            id_val.append(pat_id_new[i])
        elif pat_id_new[i] in test_id:
            x_te.append(x_new[i])
            y_te.append(y_new[i])
            id_te.append(pat_id_new[i])
        else:
            raise ValueError("ID mismatch")

    X_train=np.asarray(x_tr);y_train=np.asarray(y_tr); id_tr=np.asarray(id_tr)
    x_val=np.asarray(x_val);y_val=np.asarray(y_val); id_val=np.asarray(id_val)
    x_te=np.asarray(x_te);y_te=np.asarray(y_te); id_te=np.asarray(id_te)

    print("Train patients",len(train_id), " with number of samples:", len(x_tr), flush=True)
    print("Valid patients",len(valid_id), " with number of samples:", len(x_val), flush=True)
    print("Test patients",len(test_id), " with number of samples:", len(x_te), flush=True)

    print("----------------------------------------------", flush=True)
    print("Extract number of validation and test images:", test_val_samples, flush=True)

    # Extract correct number of validation and test images 
    val_size = 1-test_val_samples/x_val.shape[0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=np_seed)
    sss.get_n_splits(x_val, y_val)
    for index,_ in sss.split(x_val, y_val):
        X_valid = x_val[index]
        y_valid = y_val[index]

    np_seed += 1
    test_size = 1-test_val_samples/x_te.shape[0]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=np_seed)
    sss.get_n_splits(x_te, y_te)
    for index,_ in sss.split(x_te, y_te):
        X_test = x_te[index]
        y_test = y_te[index]

    # Validation Data
    val_labels = y_valid
    val_paths  = X_valid.reshape(X_valid.shape[0],1)
    # Test Data
    test_labels = y_test[0:-1]
    test_paths  = X_test.reshape(X_test.shape[0],1)[0:-1]
    # Train Data
    train_labels = y_train
    train_paths  = X_train.reshape(X_train.shape[0],1)
    print("----------------------------------------------", flush=True)


    print("CREATE DIR and SAVE CSV DATA", flush=True)
    if not os.path.isdir(tfrecord_dir):
        print("Creating directory:", tfrecord_dir, flush=True)
        os.makedirs(tfrecord_dir)
    assert(os.path.isdir(tfrecord_dir))

    tfrecord_dir_tr = tfrecord_dir+"/train"
    tfrecord_dir_vl = tfrecord_dir+"/valid"
    tfrecord_dir_te = tfrecord_dir+"/test"

    if not os.path.isdir(tfrecord_dir_tr):
        print("Creating directory:", tfrecord_dir_tr, flush=True)
        os.makedirs(tfrecord_dir_tr)
    assert(os.path.isdir(tfrecord_dir_tr))

    if not os.path.isdir(tfrecord_dir_vl):
        print("Creating directory:", tfrecord_dir_vl, flush=True)
        os.makedirs(tfrecord_dir_vl)
    assert(os.path.isdir(tfrecord_dir_vl))

    if not os.path.isdir(tfrecord_dir_te):
        print("Creating directory:", tfrecord_dir_te, flush=True)
        os.makedirs(tfrecord_dir_te)
    assert(os.path.isdir(tfrecord_dir_te))

    # Save csv files
    print("Save train.csv file to:", tfrecord_dir_tr+"/train.csv", flush=True)
    train_data = np.concatenate((train_paths,train_labels),axis=1)
    df_tr = pd.DataFrame(columns=header,data=train_data)
    df_tr.to_csv(tfrecord_dir_tr+"/train.csv", mode='w', header=True,index=False)

    print("Save validation.csv file to:", tfrecord_dir_vl+"/valid.csv", flush=True)
    val_data = np.concatenate((val_paths,val_labels),axis=1)
    df_vl = pd.DataFrame(columns=header,data=val_data)
    df_vl.to_csv(tfrecord_dir_vl+"/valid.csv", mode='w', header=True,index=False)

    print("Save test.csv file to:", tfrecord_dir_te+"/test.csv", flush=True)
    test_data = np.concatenate((test_paths,test_labels),axis=1)
    df_te = pd.DataFrame(columns=header,data=test_data)
    df_te.to_csv(tfrecord_dir_te+"/test.csv", mode='w', header=True,index=False)

    print('Train data shape (path+labels):', train_data.shape, flush=True)
    print('Validation data shape (path+labels):', val_data.shape, flush=True)
    print('Test data shape (path+labels):', test_data.shape, flush=True)
    print("----------------------------------------------", flush=True)


    print("BUILD TFRECORD FILES", flush=True)
    # Train TFR
    # --------------------------------------------
    images    = []
    for path in train_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        if image_array.shape != (512,512):
            image_array = np.asarray(image.resize((512,512)).convert("L"))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,512,512)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == train_labels.shape[0]
    print('Train images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_tr, images.shape[0], progress_interval=5000) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(train_labels[order])
    # --------------------------------------------
    # Valid TFR
    # --------------------------------------------
    images    = []
    for path in val_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        if image_array.shape != (512,512):
            image_array = np.asarray(image.resize((512,512)).convert("L"))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,512,512)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == val_labels.shape[0]
    print('Valid images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_vl, images.shape[0], progress_interval=500) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(val_labels[order])
    # --------------------------------------------
    # Test TFR
    # --------------------------------------------
    images    = []
    for path in test_paths:
        image = Image.open(path[0])
        image_array = np.asarray(image.convert("L"))
        if image_array.shape != (512,512):
            image_array = np.asarray(image.resize((512,512)).convert("L"))
        image_array = image_array / image_array.max() * 255
        image_array = image_array.astype(np.uint8)
        image_array = image_array.reshape(1,512,512)
        images.append(image_array)
    images = np.asarray(images)

    assert images.shape[0] == test_labels.shape[0]
    print('Test images shape:', images.shape, flush=True)
    with TFRecordExporter(tfrecord_dir_te, images.shape[0], progress_interval=500) as tfr:
        order = tfr.choose_shuffled_order()
        for idxi in range(order.size):
            tfr.add_image(images[order[idxi]])
        tfr.add_labels(test_labels[order])
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

    p = add_command( 'create_from_xray', 'Create dataset from xray iamges.')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')

    p = add_command( 'create_from_brain', 'Create dataset from xray iamges.')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
