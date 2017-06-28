# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import os
from scipy.misc import imread, imresize
import random
import pandas as pd
from keras.utils import np_utils
from time import time
from skimage.color import gray2rgb
from glob import glob
import hashlib

def process_train(folder, channels, img_rows, img_cols, val_size = None):
    '''Reads all the class folders in train folder, create train and
    validation datasets'''
    t_0 = time()
    
    # Get all class folders
    folder_file  = get_folders(folder)
    
    nb_classes = len(folder_file)
    print 'Number of classes in training:', nb_classes
    total_images = get_image_count(folder_file)
    print 'Total no. of images in training:', total_images
    
    # Assign numbers to class names
    class_names = os.listdir(folder)
    class_labels = {}
    for i, j in enumerate(class_names):
        class_labels[j] = i
    
    # Read train files and create train and validation set    
    if val_size is None:
        train_items = list(xrange(total_images))
        random.shuffle(train_items)
        
        X_train = np.ndarray(shape = (total_images, channels, img_rows, img_cols), dtype = np.float32)
        y_train = np.zeros(total_images)
        
        image_count = 0
        
        for inx, sub_folder in enumerate(folder_file):
            print 'Processing class:', inx, ',' , 'class_names[inx]'
            image_files = process_class_folder(sub_folder)
            for image in image_files:
                image_data = process_image(sub_folder, image, img_rows, img_cols)
                position = int(train_items.index(int(image_count)))
                X_train[position, :, :, :] = image_data
                y_train[position] = inx
                image_count += 1
        
        y_train = np_utils.to_categorical(y_train, nb_classes)
                
    else:
        train_items = list(xrange(total_images))
        random.shuffle(train_items)
        val_image_size = int(total_images * val_size)
        val_items = train_items[:val_image_size]
        train_items = train_items[val_image_size:]
        
        X_train = np.ndarray(shape = (len(train_items), channels, img_rows, img_cols), dtype = np.float32)
        X_val = np.ndarray(shape = (len(val_items), channels, img_rows, img_cols), dtype = np.float32)
        y_train = np.zeros(len(train_items))
        y_val = np.zeros(len(val_items))
        
        image_count = 0
        
        for inx, sub_folder in enumerate(folder_file):
            print 'Processing class:', inx, ',' , class_names[inx]
            image_files = process_class_folder(sub_folder)
            for image in image_files:
                image_data = process_image(sub_folder, image, img_rows, img_cols)
                if image_count in train_items:
                    position = int(train_items.index(int(image_count)))
                    X_train[position, :, :, :] = image_data
                    y_train[position] = inx
                else:
                    position = int(val_items.index(int(image_count)))
                    X_val[position, :, :, :] = image_data
                    y_val[position] = inx
                image_count += 1
        
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_val = np_utils.to_categorical(y_val, nb_classes)
    t_1 = time() - t_0
    print 'Time taken:', t_1                    
    
    if val_size is None:
        print 'Completed Processing'
        print 'Size of training set:', X_train.shape
        print 'Size of training labels:', y_train.shape
        return X_train, y_train, class_labels
    else:
        print 'Completed Processing'
        print 'Size of training set:', X_train.shape
        print 'Size of training labels:', y_train.shape
        print 'Size of validation set:', X_val.shape
        print 'Size of validation labels:', y_val.shape
        return X_train, y_train, X_val, y_val, class_labels
    
    
def get_folders(folder):
    '''Returns all subfolders in a given folder as a list'''
    folder_file = []
    for folders in os.listdir(folder):
        sub_folder = os.path.join(folder, folders)
        folder_file.append(sub_folder)
    return folder_file

def get_image_count(folder_file):
    ''' Gives the count of images inside a folder'''
    total_images = 0
    for sub_folder in folder_file:
        total_images += len(os.listdir(sub_folder))
    return total_images

def process_class_folder(sub_folder):
    '''Returns all the image names inside a folder'''
    image_files = os.listdir(sub_folder)
    return image_files
    
def process_image(sub_folder, image, img_rows, img_cols):
    '''Process individual images'''
    
    # Using the mean pixel values used in VGG16 Model
    mean_pixel = [103.939, 116.779, 123.68]
    image_path = os.path.join(sub_folder, image)
    image_file = imread(image_path)
    
    # If the image has 4 channels, change it to 3 channels
    if len(image_file.shape) > 2 and image_file.shape[2] == 4:
        image_file = image_file[:,:,:-1]
    # If the image is in grey scale, change it to RGB
    if len(image_file.shape) < 3:
        image_file = gray2rgb(image_file)
    image_file = image_file.astype(np.float32, copy=False)
    # There are some images where the actual image we are interested is in 
    # the right side. For such images remove the left half
    if image_file.shape[1] > image_file.shape[0]:
        new_shape = image_file.shape[1] / 2
        image_file = image_file[:,new_shape:,:]
    # one more image pattern
    elif image_file.shape[0] == 1000 & image_file.shape[1] == 677:
        image_file = image_file[:,:455,:]
    image_resized = imresize(image_file, (img_rows, img_cols))
    # normalize the image
    for c in xrange(3):
        image_resized[:, :, c] = image_resized[:, :, c] - mean_pixel[c]
        image_res = image_resized.transpose((2,0,1))
    return image_res
    
def predict_test(test_folder, model, img_rows, img_cols, class_labels):
    '''Makes predictions on the files in test folder and returns the
    submission file'''
    image_files = os.listdir(test_folder)
    no_of_files = len(image_files)
    columns = ['id','Mobile_Theme']
    index = xrange(0,no_of_files)
    submission = pd.DataFrame(index = index, columns = columns)

    for image_count, image in enumerate(image_files):
        image_data = process_image(test_folder, image, img_rows, img_cols)
        image_res = np.expand_dims(image_data, axis = 0)
        pred = model.predict(image_res, verbose = 0)[0]
        pred = list(pred)
        predic = pred.index(max(pred))
        submission['id'][image_count] = image
        submission['Mobile_Theme'][image_count] = class_labels.keys()[class_labels.values().index(predic)]

        if image_count % 100 == 0:
            print 'Processing test image', image_count, 'of', no_of_files
    return submission
    
def clean_train(train_folder):
    '''Removes duplicates in train folder where the same images appears in
    more than one class'''
    hashes = {}
    labels = {}

    print 'computing md5 of training data'

    for fname in glob(train_folder+'/*/*.jpg'):
        labels[fname] = fname.split('\\')[-2]
        h = hashlib.md5(open(fname,'rb').read()).hexdigest()  
        if h in hashes:
            hashes[h].append(fname)
        else:
            hashes[h] = [fname]
    
    # Find duplicates
    repeated = sum(1 for k,v in hashes.items() if len(v) > 1 )
    print 'Files appearing more than once in train:', repeated
    
    del_files = []
    
    # Find duplicate images with different class names
    for k,v in hashes.items():
        if len(v) > 1:
            c = set([labels[x] for x in v])
            if len(c) > 1:
                del_files = del_files + v
    
    for x in del_files:
        os.remove(x)

    print len(del_files), 'images deleted from training set'
    
def find_leak(train_folder, test_folder):
    '''Finds images present in both training and test set'''

    hashes = {}
    labels = {}

    print 'computing md5 of training data'

    for fname in glob(train_folder+'/*/*.jpg'):
        labels[fname] = fname.split('\\')[-2]
        h = hashlib.md5(open(fname,'rb').read()).hexdigest()  
        if h in hashes:
            hashes[h].append(fname)
        else:
            hashes[h] = [fname]

    print 'comparing training and test set'
    
    leaks = []
    for fname in glob(test_folder+'/*.jpg'):
        h = hashlib.md5(open(fname,'rb').read()).hexdigest()
        if h in hashes:
            leaks.append((fname.split('\\')[-1],hashes[h][0].split('//')[-2]))

    print 'Number of test images present in train:{}'.format(len(leaks))
    return leaks