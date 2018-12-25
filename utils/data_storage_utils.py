'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
you can ignore this class completely.

This class is solely created to store the loaded and transformed data for project 1, 
wind speed and direction detection. 

It allows us to access the data without re-loading and re-ransorming them.

'''

import os.path
import pandas as pd
import numpy as np
from transformers.featureGenerator import FeatureGenerator
from sklearn.model_selection import train_test_split
from utils.data_processing_utils import *

# The values below are used to constructed the path of data files
project = 'project1'
data_path = 'data/transformed_data/'
suffix_train_data = '_transformed_train.csv'
suffix_test_data = '_transformed_test.csv'

# We specify the default window size and packets number.
window_size = 1
num_packets = 6

# This ratio gives us the last 6400 test values in loaded data.
test_size = 0.16665

# initialize class label, "" means front wind
directions = ["", "away", "left", 'right']

class DataContianer():
	"""A class used to store loaded and transformed data.

	Attributes:
	drone_name: str
	 	the name of a drone, we have 4 drones in total
	sensor: str
		the name of which sensor the data is generated from
	num_classes: int
		the number of class labels
	reduce_noise: boolean
		true if apply Fast Fourier Transform to reduce white noise
		(default is False)
	k: int
		the number of the top-k dominant powers,
		only specified when reduce_noise is true (default is 0).
	is_directional: boolean
		true if wind direction detection,
		false if wind speed detection (default is False)
	"""
	def __init__(self, drone_name, sensor, num_classes, reduce_noise=False, k=0, is_directional=False):
		"""
		Parameters
		----------
		drone_name: str
		 	the name of a drone, we have 4 drones in total
		sensor: str
			the name of which sensor the data is generated from
		num_classes: int
			the number of class labels
		reduce_noise: boolean
			true if apply Fast Fourier Transform to reduce white noise
			(default is False)
		k: int
			the number of the top-k dominant powers,
			only specified when reduce_noise is true (default is 0).
		is_directional: boolean
			true if wind direction detection,
			false if wind speed detection (default is False)
		"""
		self.drone_name = drone_name
		self.sensor = sensor
		self.num_classes = num_classes
		self.is_directional = is_directional
		self.reduce_noise = reduce_noise
		self.k = k
		self._set_train_test_data()

	def _load_data(self, wind_level, num_packets, direction=""):
		"""Load data files of specified sensor into a dataframe

		Parameters
		----------
		wind_level: int
		 	wind speed
		num_packets: int
			then name of data packets to load
		num_classes: int
			the number of class labels
		direction: str
			the direction of where the wind is coming from (default is "")
		
		Returns
		-------
		ndarray
			the data of specified sensor for the specified wind speed or wind direction.
		"""
		data = load_data(wind_level, num_packets, project, self.drone_name, direction)
		data = separate_data_based_on_apparatus(data)
		return data[self.sensor]

	def _set_train_test_data(self):
		"""Splits data into train and test data and stores them.

		Parameters
		----------
		
		Returns
		-------
		"""
		X_train, X_test = pd.DataFrame(), pd.DataFrame()
		y_train, y_test = [], []

		# load all data of a specific sensor
		for label in range(self.num_classes):
			wind_level = -1
			direction = ""
			if self.is_directional:
				wind_level = 2.5
				direction = directions[label]
			else:
				wind_level = label

			X = self._load_data(wind_level, num_packets, direction)
			y = [label for x in range(X.shape[0])]		

			X_train_temp, X_test_temp, y_train_temp, y_test_temp = \
			train_test_split(X, y, test_size=test_size, shuffle=False)

			X_train = X_train.append(X_train_temp)
			y_train.append(y_train_temp)
			X_test = X_test.append(X_test_temp)
			y_test.append(y_test_temp)

		self.X_train = X_train
		self.y_train = np.array(y_train).flatten()

		self.X_test = X_test
		self.y_test = np.array(y_test).flatten()

		# generate file paths
		directional = '_directional' if self.is_directional else ""
		noise_reduced = '_fft_'+str(self.k) if self.reduce_noise else ""

		train_data_path = data_path+self.drone_name+"/"+\
						self.sensor+directional+noise_reduced+suffix_train_data
		
		# transform training data and store the transform data
		self.X_train_transformed, self.y_train_transformed = \
		self._transform_data(self.X_train, self.y_train, train_data_path)

		# transform test data and store the transform data
		test_data_path = data_path+self.drone_name+"/"+\
						self.sensor+directional+noise_reduced+suffix_test_data

		self.X_test_transformed, self.y_test_transformed = \
		self._transform_data(self.X_test, self.y_test, test_data_path)

	def _transform_data(self, X, y, file_path):
		"""Transforms data and stores them into a file if a file does not already exist.

		Parameters
		----------
		X: pandas data frame
			raw data

		y: ndarray
			raw data labels

		Returns
		X_transformed: pandas data frame
			tranformed data
		y_transformed: ndarray
			tranformed data labels
		-------
		"""		

		X_transformed, y_transformed = None, None
		if os.path.isfile(file_path):
			# load data from file if the file exists
			df = pd.read_csv(file_path)
			X_transformed = df.iloc[:, :-1]
			y_transformed = df.iloc[:, -1]
		else:
			X_transformed, y_transformed = self._generate_features(X, y)
			self._save_to_file(X_transformed, y_transformed, file_path)

		return X_transformed, y_transformed

	def _generate_features(self, X, y):
		"""Sends raw data into a pipeline and returns transformed data.

		Parameters
		----------
		X: pandas data frame
			raw data

		y: ndarray
			raw data labels

		Returns
		X_transformed: pandas data frame
			tranformed data
		y_transformed: ndarray
			tranformed data labels
		-------
		"""		
		feature_generator = FeatureGenerator(window_size, self.sensor, self.reduce_noise, self.k)
		feature_generator.fit(X, self.num_classes)
		X_transformed = feature_generator.transform(X)
		y_transformed = self.adjust_label_amount(y, self.num_classes)

		return X_transformed, y_transformed

	def _save_to_file(self, X, y, file_path):
		"""Helper function, saves tranformed data to a file

		Parameters
		----------
		X: pandas data frame
			tranformed data

		y: ndarray
			tranformed data labels

		Returns
		-------
		"""	
		df = X
		df['label'] = y
		df.to_csv(file_path, index=False, sep=',')

	def adjust_label_amount(self, y, num_classes):
		"""Adjust the number of labels of all classes
		
		After performing feature transformation on the training data,
		the amount of data points will decrease becasue of the usage of sliding window.
		As a result, we need to reduce the number of labels for each class, accordingly.

		Parameters
		----------
		y: ndarray
			raw data labels

		Returns
		ndarray
			 data labels after reduced quantity
		-------
		"""

			
	    rows = y.shape[0]
	    rows_needed = int(sliding_window * 1000 / 10)

	    if rows_needed > rows:
	        print('Not enough data.')
	        return None

	    # calculate how many rows for one class
	    num_data_points = rows//num_classes
	    remainder = num_data_points % rows_needed
	    counter = 0
	    for k in range(num_data_points):
	        if k + rows_needed <= num_data_points:
	            counter += 1
	        else:
	            break
	            
	    if remainder / rows_needed > 0.9:
	        counter += 1

	    # generate new labels
	    y_new = []
	    for c in range(num_classes):
	        label = [c for x in range(counter)]
	        y_new.append(label)

	    return np.array(y_new).flatten()

	def get_transformed_train_test_data(self):
		"""Returns data in the form of traning data test sets

		Parameters
		----------

		Returns
		X_train_transformed: pandas data frame
			tranformed data, training set
		y_train_transformed: ndarray
			 tranformed data labels, training set
		X_test_transformed: pandas data frame
			tranformed data, test set
		y_test_transformed: ndarray
			 tranformed data labels, test set
		-------
		"""			
		return (self.X_train_transformed, self.y_train_transformed,
		 		self.X_test_transformed, self.y_test_transformed)

