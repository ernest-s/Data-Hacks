import numpy as np
import os
from scipy.misc import imread, imsave, imresize, imshow
import random
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from sklearn.cross_validation import train_test_split
from time import time

# get file list inside train folder
def get_file_list(folder):
    folder_file = []
    for folders in os.listdir(folder):
        sub_folder = os.path.join(folder, folders)
        folder_file.append(sub_folder)
    return folder_file

# function to separate training set by flight phases
def partition_data(folder, dest_folder):
	# dest_folder is the folder where the partitioned data will be stored
	# dest_folder should have 8 folders inside it
	# the folder names should be 0,1,2... 8
	# function will split phase 0 data into 2 partitions, phase 1, 3, 7 
	# into a single partition, phase 2, 4, 6 into 4 partitions and phase 5
	# into 5 partitions
	time_0 = time()

	folder_file = get_file_list(folder)
	print 'Found', len(folder_file), 'files'

	col_df = pd.read_csv(folder_file[0])
	column_names = list(col_df.columns.values)

	X_train = pd.DataFrame(columns = column_names)
	
	for i in xrange(0,8):
		print 'Processing phase', i

		ph_count = 0

		for s_no, sub_folder in enumerate(folder_file):
			if (s_no + 1) % 50 == 0:
				print 'Processing file', s_no +1, 'of', len(folder_file)
			df = pd.read_csv(sub_folder)
			df = df[df.PH == i]
			X_train = X_train.append(df)
			ph_count += df.shape[0]

			if i == 0:
				if (ph_count == 417026 and len(os.listdir('train_part\\0')) == 1):
					file_name = os.path.join(dest_folder, '0', 'X_train_0_b.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 200000 and len(os.listdir('train_part\\0')) == 0):
					file_name = os.path.join(dest_folder, '0', 'X_train_0_a.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 1:
				if (ph_count == 52395 and len(os.listdir('train_part\\1')) == 0):
					file_name = os.path.join(dest_folder, '1', 'X_train_1.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 2:
				if (ph_count == 1186587 and len(os.listdir('train_part\\2')) == 3):
					file_name = os.path.join(dest_folder, '2', 'X_train_2_d.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 900000 and len(os.listdir('train_part\\2'))==2):
					file_name = os.path.join(dest_folder, '2', 'X_train_2_c.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 600000 and len(os.listdir('train_part\\2'))==1):
					file_name = os.path.join(dest_folder, '2', 'X_train_2_b.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 300000 and len(os.listdir('train_part\\2'))==0):
					file_name = os.path.join(dest_folder, '2', 'X_train_2_a.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 3:
				if (ph_count == 109020 and len(os.listdir('train_part\\3')) == 0):
					file_name = os.path.join(dest_folder, '3', 'X_train_3.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 4:
				if (ph_count == 1167633 and len(os.listdir('train_part\\4')) == 3):
					file_name = os.path.join(dest_folder, '4', 'X_train_4_d.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 900000 and len(os.listdir('train_part\\4'))==2):
					file_name = os.path.join(dest_folder, '4', 'X_train_4_c.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 600000 and len(os.listdir('train_part\\4'))==1):
					file_name = os.path.join(dest_folder, '4', 'X_train_4_b.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 300000 and len(os.listdir('train_part\\4'))==0):
					file_name = os.path.join(dest_folder, '4', 'X_train_4_a.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 5:
				if (ph_count == 1962381 and len(os.listdir('train_part\\5')) == 4):
					file_name = os.path.join(dest_folder, '5', 'X_train_5_e.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 1600000 and len(os.listdir('train_part\\5'))==3):
					file_name = os.path.join(dest_folder, '5', 'X_train_5_d.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 1200000 and len(os.listdir('train_part\\5'))==2):
					file_name = os.path.join(dest_folder, '5', 'X_train_5_c.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 800000 and len(os.listdir('train_part\\5'))==1):
					file_name = os.path.join(dest_folder, '5', 'X_train_5_b.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 400000 and len(os.listdir('train_part\\5'))==0):
					file_name = os.path.join(dest_folder, '5', 'X_train_5_a.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			elif i == 6:
				if (ph_count == 1196532 and len(os.listdir('train_part\\6')) == 3):
					file_name = os.path.join(dest_folder, '6', 'X_train_6_d.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 900000 and len(os.listdir('train_part\\6'))==2):
					file_name = os.path.join(dest_folder, '6', 'X_train_6_c.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 600000 and len(os.listdir('train_part\\6'))==1):
					file_name = os.path.join(dest_folder, '6', 'X_train_6_b.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
				elif (ph_count > 300000 and len(os.listdir('train_part\\6'))==0):
					file_name = os.path.join(dest_folder, '6', 'X_train_6_a.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
			else:
				if (ph_count == 22190 and len(os.listdir('train_part\\7')) == 0):
					file_name = os.path.join(dest_folder, '7', 'X_train_7.csv')
					X_train.to_csv(file_name, index = False)
					X_train = pd.DataFrame(columns = column_names)
	
	time_taken = time() - time_0
	print 'Completed processing'
	print "Time taken: %f sec" % (time_taken)

	return