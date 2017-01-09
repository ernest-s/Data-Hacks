import numpy as np
import os
from scipy.misc import imread, imsave, imresize, imshow
import random
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy
import sys
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from math import sqrt

from skimage import color
from skimage import io
from skimage.color import rgb2gray
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import model_from_json

from keras import backend as K
K.set_image_dim_ordering('th')

class ErnestNet(object):

  def __init__(self):
    pass

  def get_image_names(self, folder):
    image_file = os.listdir(folder)
    image_file = [i for i in image_file if 'mask' not in i]
    return image_file

  def process_image(self, image_path):
    image_file = imread(image_path)
    image_size = image_file.shape
    image_resized = imresize(image_file, (128, 128))
    image_transposed = np.asarray (image_resized.transpose(2,0,1), dtype = 'float32')
    image_data = image_transposed / 255
    return image_data, image_size
	
  def process_mask(self, y_image_path):
    image_file = imread(y_image_path)
    image_resized = imresize(image_file, (64,64))
    image_resized = np.where(image_resized > 0, 1, 0)
    return image_resized.flatten()

  def process_train_data(self, folder, image_file):
    self.X_train = np.ndarray(shape = (len(image_file), 3, 128, 128), dtype = np.float32)
    self.y_train = np.zeros(shape=(len(image_file), 4096))
    for i, image_name in enumerate(image_file):
      image_path = os.path.join(folder, image_name)
      self.X_train[i,:,:,:], _ = self.process_image(image_path)
      y_image_path = image_path[:-4] + '-mask.jpg'
      self.y_train[i,:] = self.process_mask(y_image_path)
 
  def process_train(self, train_folder):
    train_image_file = self.get_image_names(train_folder)
    self.process_train_data(train_folder, train_image_file)
    print 'X_train shape:', self.X_train.shape
    print 'y_train shape:', self.y_train.shape
      
  def create_network(self, channels = 3, image_rows = 128, image_cols = 128, lr = 0.01, decay = 1e-6, momentum = 0.9):
    self.model = Sequential()
 
    self.model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape = (channels, image_rows, image_cols)))
    self.model.add(Activation('relu'))
    self.model.add(Convolution2D(32, 3, 3))
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))
 
    self.model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    self.model.add(Activation('relu'))
    self.model.add(Convolution2D(64, 3, 3))
    self.model.add(Activation('relu'))
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.25))
 
    self.model.add(Flatten())
    self.model.add(Dense(4096))
    self.model.add(Activation('sigmoid'))
    
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)
    
    self.model.compile(loss='mse', optimizer=sgd)
    print 'Model created'
    
  def train_model(self, nb_epoch = 50, batch_size = 64, validation_split = 0.1, verbose = 1):
    self.model.fit(self.X_train, self.y_train, nb_epoch = nb_epoch, batch_size = batch_size, validation_split = validation_split, verbose = verbose)

  def save_model(self, file_name):
    json_model = self.model.to_json()
    weights_name = file_name + '_weights.h5'
    file_name = file_name + '.json'
    with open(file_name, "w") as json_file:
      json_file.write(json_model)
    self.model.save_weights(weights_name)

  def load_model(self, model_name, model_weights):
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    model.load_weights(model_weights)

  def predict_test(self, test_folder, destination_folder):
    test_image_file = self.get_image_names(test_folder)
    for i, im in enumerate(test_image_file):
      print 'Processing Test Image:', i
      file_name = os.path.join(test_folder, im)
      image = imread(file_name)
      im_final = self.apply_gaussian_filter(image)
      image_gray = rgb2gray(im_final)
      blobs = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.55)
      blob_list = self.process_blobs(image, blobs)
      self.create_mask(image, im, blob_list, destination_folder)	  
	
  def apply_gaussian_filter(self,image):
    im_g = gaussian_filter(image, 3)
    im_norm = (im_g - im_g.min()) / (float(im_g.max()) - im_g.min())
    im_norm[im_norm < 0.5] = 0
    im_norm[im_norm >= 0.5] = 1
    im_final = 255 - (im_norm * 255).astype(np.uint8)
    return im_final

  def process_blobs(self, image, blobs):
    blob_list = []
    if len(blobs) > 0:
      for blob in blobs:
        x = blob[0]
        y = blob[1]
        length = np.ceil(blob[2] * 4 / sqrt(2))
        if (blob[2] > 10):
          blob_list.append([x, y, length])	  
      if len(blob_list) > 0:
        return blob_list
      else:
        return None
    else:
      return None

  def create_mask(self, image, im, blob_list, destination_folder):
    mask_file = np.zeros(shape = (image.shape[0],image.shape[1]))
    if blob_list is not None:
      for bl in blob_list:
        x1 = int(bl[0] - bl[2])
        y1 = int(bl[1] - bl[2])
        x2 = int(bl[0] + bl[2])
        y2 = int(bl[1] + bl[2])
        x1 = np.max([x1, 0])
        y1 = np.max([y1, 0])
        x2 = np.min([x2, int(mask_file.shape[0])])
        y2 = np.min([y2, int(mask_file.shape[1])])
        image1 = image[x1:x2, y1:y2, :]
        im1_sh = image1.shape
        image1 /= 255
        image1_resized = imresize(image1, (128, 128))
        image1_transposed = np.asarray (image1_resized.transpose(2,0,1), dtype = 'float32')
        final_data = np.ndarray(shape = (1,3,128,128))
        final_data[0,:,:,:] = image1_transposed
        y_pred = self.model.predict(final_data, verbose = 0)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        y_pred = np.reshape(y_pred, (64,64))
        y_pred = imresize(y_pred, im1_sh)
        mask_file[x1:x2, y1:y2] += y_pred
        mask_file = np.where(mask_file > 0, 255, 0)
    final_path = destination_folder + '/' + im[:-4] + '-mask.jpg'
    cv2.imwrite(final_path, mask_file)