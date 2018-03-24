# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 00:37:26 2018

@author: Ernest Kirubakaran Selvaraj
"""

import os
import hashlib
import numpy as np
from glob import glob
from random import shuffle
from keras import optimizers
from skimage.color import gray2rgb
from scipy.misc import imread, imsave
from keras.applications import InceptionV3
from keras.models import Model, Sequential
from keras.layers.core import Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Stop training if validation loss doesn't improve for 10 epochs
earlystop = EarlyStopping(monitor = "val_loss", 
                          patience = 10, 
                          verbose = 1, 
                          mode = "auto")

# Save the best model after every epoch
checkpoint = ModelCheckpoint(filepath = "inceptionv3.hdf5", 
                             verbose = 1, 
                             save_best_only = True)

# Reduce the learning rate after validation loss plateaus
reducelr = ReduceLROnPlateau(monitor = "val_loss", 
                             factor = 0.2,
                             patience = 5)

TARGET_SIZE = (299, 299) # Input shape for Inception v3
BATCH_SIZE = 32 # Batch size for training

def clean_train(train_folder):
    '''Removes duplicates in train folder where the same images appears in
    more than one class'''
    hashes = {}
    labels = {}

    print("computing md5 of training data")

    for fname in glob(train_folder+"/*/*.jpg"):
        labels[fname] = fname.split("//")[-2]
        h = hashlib.md5(open(fname,"rb").read()).hexdigest()  
        if h in hashes:
            hashes[h].append(fname)
        else:
            hashes[h] = [fname]
    
    # Find duplicates
    repeated = sum(1 for k,v in hashes.items() if len(v) > 1 )
    print("Files appearing more than once in train: ", repeated)
    
    del_files = []
    
    # Find duplicate images with different class names
    for k,v in hashes.items():
        if len(v) > 1:
            c = set([labels[x] for x in v])
            if len(c) > 1:
                del_files = del_files.append(v)
    
    for x in del_files:
        os.remove(x)

    print(len(del_files), "images deleted from training set")
    
def find_leak(train_folder, test_folder):
    '''Finds images present in both training and test set'''

    hashes = {}
    labels = {}

    print("computing md5 of training data")

    for fname in glob(train_folder+"/*/*.jpg"):
        labels[fname] = fname.split("//")[-2]
        h = hashlib.md5(open(fname,"rb").read()).hexdigest()  
        if h in hashes:
            hashes[h].append(fname)
        else:
            hashes[h] = [fname]

    print("comparing training and test set")
    
    leaks = []
    for fname in glob(test_folder+"/*.jpg"):
        h = hashlib.md5(open(fname,"rb").read()).hexdigest()
        if h in hashes:
            leaks.append((fname.split("//")[-1],hashes[h][0].split("//")[-2]))

    print("Number of test images present in train:{}".format(len(leaks)))
    return leaks

def process_train_images(train_folder):
    ''' Function to convert training images to 3 channels (for images having
    4 channels or less than 3 channels)''' 
    
    classes = os.listdir(train_folder)
    for cla in classes:
        cla_path = os.path.join("dataset", "train", cla)
        for img in os.listdir(cla_path):
            img_path = os.path.join("dataset", "train", cla, img)
            img_file = imread(img_path)
            if len(img_file.shape) < 3:
                img_file = gray2rgb(img_file)
                img_file = img_file.astype(np.float32, copy = False)
                imsave(img_path, img_file)
            if len(img_file.shape) == 4:
                img_file = img_file[:,:,:-1]
                img_file = img_file.astype(np.float32, copy = False)
                imsave(img_path, img_file)
                
def process_test_images(test_folder):
    ''' Function to convert test images to 3 channels (for images having
    4 channels or less than 3 channels)'''
    for img in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img)
        img_file = imread(img_path)
        if len(img_file.shape) < 3:
            img_file = gray2rgb(img_file)
            img_file = img_file.astype(np.float32, copy = False)
            imsave(img_path, img_file)
        if len(img_file.shape) == 4:
            img_file = img_file[:,:,:-1]
            img_file = img_file.astype(np.float32, copy = False)
            imsave(img_path, img_file)
                
# Pre-processing function for Inception v3 model
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def remove_percentage(list_a, percentage):
    ''' Function to randomly pick x percentage from a list'''
    shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: 
        return []
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b

def create_val_set(val_size):
    '''Function to create a validation set from training images'''
    if not os.path.exists("dataset//valid"):
        os.makedirs("dataset//valid")
    class_list = os.listdir("dataset//train")
    for cla in class_list:
        if os.path.exists(os.path.join("dataset", "valid", cla)):
            if len(os.listdir(os.path.join("dataset", "valid", cla))) == 0:
                new_files = os.listdir(os.path.join("dataset", "train", cla))
                new_files = remove_percentage(new_files, val_size)
                for nf in new_files:
                    os.rename(os.path.join("dataset", "train", cla, nf), 
                              os.path.join("dataset", "valid", cla, nf))
            else:
                new_files = os.listdir(os.path.join("dataset", "valid", cla))
                for nf in new_files:
                    os.rename(os.path.join("dataset", "valid", cla, nf),
                              os.path.join("dataset", "train", cla, nf))
                new_files = os.listdir(os.path.join("dataset", "train", cla))
                new_files = remove_percentage(new_files, val_size)
                for nf in new_files:
                    os.rename(os.path.join("dataset", "train", cla, nf), 
                              os.path.join("dataset", "valid", cla, nf))
        else:
            os.makedirs(os.path.join("dataset", "valid", cla))
            new_files = os.listdir(os.path.join("dataset", "train", cla))
            new_files = remove_percentage(new_files, val_size)
            for nf in new_files:
                os.rename(os.path.join("dataset", "train", cla, nf), 
                          os.path.join("dataset", "valid", cla, nf))

def define_model():
    ''' Load a pre-trained inception V3 model and change the top layers to 
    match the number of classes of our problem'''
    base_model = InceptionV3(weights = "imagenet", 
                         include_top = False, 
                         input_shape = (299, 299, 3))
    for i in range(len(base_model.layers)):
        base_model.layers[i].trainable = False
    add_model = Sequential()
    add_model.add(Flatten(input_shape = base_model.output_shape[1:]))
    add_model.add(Dense(256, activation = "relu"))
    add_model.add(Dense(len(os.listdir("dataset//train")), activation="softmax"))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss = "categorical_crossentropy", 
              optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9),
              metrics = ["accuracy"])
    model.summary()
    return(model)