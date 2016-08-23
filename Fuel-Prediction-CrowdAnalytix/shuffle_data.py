import numpy as np
import os
from scipy.misc import imread, imsave, imresize, imshow
import random
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from sklearn.cross_validation import train_test_split
from time import time

# get zip file list inside train folder
def get_folders(folder):
    folder_file = []
    for folders in os.listdir(folder):
        sub_folder = os.path.join(folder, folders)
        folder_file.append(sub_folder)
    return folder_file

# get list of csv files inside a zip file
def get_files(sub_folder):
	file_list = zipfile.ZipFile(sub_folder).namelist()
	zfile = zipfile.ZipFile(sub_folder)
	return file_list, zfile

# read csv file inside zip folder
def read_csv_zip(csv_file, zfile):
	load_df = pd.read_csv(zfile.open(csv_file))
	return load_df


# shuffle the data
def shuffle_data(folder, train_destination):
	time_0 = time()

	folder_file = get_folders(folder)
	print 'Found', len(folder_file), 'folders'
	
	for s_no, sub_folder in enumerate(folder_file):
		print 'Processing folder', s_no+1, 'of', len(folder_file)
		file_list, zfile = get_files(sub_folder)

		for s_no_1, csv_file in enumerate(file_list):
			if (s_no_1+1) % 50 == 0:
				print 'Processing file', s_no_1+1, 'of', len(file_list)
			load_df = read_csv_zip(csv_file, zfile)
			load_df = load_df.sample(frac=1).reset_index(drop=True)
			file_name = os.path.join(train_destination, csv_file)
			load_df.to_csv(file_name, index = False)
	
	time_taken = time() - time_0
	print "Time taken: %f sec" % (time_taken)
	
	return
def counter_ph(folder):
	counts = [0,0,0,0,0,0,0,0]
	folder_file = []
	for files in os.listdir(folder):
		file_name = os.path.join(folder, files)
		folder_file.append(file_name)
	for i, file_name in enumerate(folder_file):
		df = pd.read_csv(file_name)
		counter = df['PH'].value_counts()
		counter_keys = counter.keys().tolist()
		for keys in counter_keys:
			counts[keys] = counts[keys] + counter[keys]
		if (i+1) % 50 == 0:
			print 'Processing file', i+1
	return counts