'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
This class is a customized feature generator extended from the base
estimator of sklearn. 

It is mainly used to generate all the high-level features for
this particular project.

'''

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class FeatureGenerator(BaseEstimator, TransformerMixin):
	"""
	A class used to transform raw sensory data and generate features.

	Attributes:
	sliding_window: int
	 	the size of sliding window
	sensor: str
		the name of which sensor the data is generated from
	reduce_noise: boolean
		true if apply Fast Fourier Transform to reduce white noise
		(default is False)
	k: int
		the number of the top-k powers,
		only specified when reduce_noise is true (default is 0).
	"""

	def __init__(self, sliding_window, sensor, reduce_noise=False, k=0):
		"""
		Parameters
		----------
		sliding_window: str
		 	the size of sliding window
		sensor: str
			the name of which sensor the data is generated from
		reduce_noise: boolean
			true if apply Fast Fourier Transform to reduce white noise
			(default is False)
		k: int
			the number of the top-k powers,
			only specified when reduce_noise is true (default is 0).
		"""
		self.sliding_window = sliding_window
		self.sensor = sensor
		self.reduce_noise = reduce_noise
		self.k = k

	def get_avg_resultant_acc(self, data):
		""" Generates average resultant acceleration for a window of data points

		Parameters
		----------
		data: panda data frame
			raw sensory data without the label

		Returns:
		avg_resultant_acc: ndarray
			the average resultant acceleration for a window of data points
		"""
		pwd = np.power(data, 2)
		sum_xyz = np.sum(pwd, 1)
		sqrt_xyz = np.sqrt(sum_xyz)
		sum_resultant_acc = np.sum(sqrt_xyz)
		avg_resultant_acc = sum_resultant_acc/100

		return avg_resultant_acc

	def get_binned_distribution_for_one_axis(self, data):
		""" Generates the binned distribution for a window of data points for one axis
		
		Parameters
		----------
		data: panda data frame
			raw sensory data without the label for one axis

		Returns:
		np.bincount(binned_data): ndarray
			the binned distribution for a window of data points for one axis
		"""
	    max_val = data.max()
	    min_val = data.min()
	    diff = max_val - min_val
	    bin_size = diff/10
	    
	    splits = [min_val+i*bin_size for i in range(0, 11)]
	    splits[0] -= 1
	    splits[-1] += 1
	    binned_data = pd.cut(data, splits, right=True, labels=False)

	    return np.bincount(binned_data)

	def get_binned_distribution(self, data):
		""" Assembles the binned distribution 
		
		Generates binned distribution for a window of data points.
		The data is consisted of values for three axes.

		Parameters
		----------
		data: panda data frame
			raw sensory data without the label

		Returns:
		np.array(results).flatten(): ndarray
			the binned distribution for a window of data points for all axis
		"""
		results = []
		for axis in data:
			result = self.get_binned_distribution_for_one_axis(data[axis])
			assert (result.sum() == data.shape[0])
			results.append(result)

		return np.array(results).flatten()

	def get_features(self, data):
		"""Aggregates and returns all the generated features 
		
		Each function is applied to one sliding window of data points

		Parameters
		----------
		data: panda data frame
			raw sensory data without the label

		Returns:
		features: pandas data frame
			tranformed data for one sliding window
		"""

		features = np.array([])

		mu = data.mean()
		features = np.hstack((features, np.array(mu)))

		std = data.std()
		features = np.hstack((features, np.array(std)))

		avg_resultant_acc = self.get_avg_resultant_acc(data)
		features = np.append(features, avg_resultant_acc)

		binned_distribution = self.get_binned_distribution(data)
		features = np.hstack((features, binned_distribution))

		#mean absoutle difference
		mad = data.mad()
		features = np.hstack((features, np.array(mad)))

		return features

	def generate_features(self, data):
		"""Generates high-level features for the raw seneory data

		Parameters
		----------
		data: panda data frame
			raw sensory data without the label

		Returns:
		np.array(final_data): ndarray
			transformed data for the entire data set
		"""

		rows, _ = data.shape
		final_data = []
		
		# Data is generated at 100Hz. Thus, a window of size x has x*100 data points,
		# which is equivalent to how many rows needed to generate one piece of data.
		rows_needed = int(self.sliding_window * 100)
		if rows_needed > rows:
			print('Not enough data.')
			return None

		num_data_points = data.shape[0]
		remainder = num_data_points % rows_needed
		# transforming features starts from here
		for i in range(num_data_points):
			if i + rows_needed <= num_data_points:
				data_in_window = data.iloc[i:(rows_needed+i), :]

				# reduce noise first if it is true
				if self.reduce_noise:
					data_in_window = self.get_top_k_power_data(data_in_window, self.k)
				transformed_features = self.get_features(data_in_window)
				final_data.append(transformed_features)
			else:
				break
		        
		# transform the remaining data points if there are enough data points
		if remainder / rows_needed > 0.9:
			data_in_window = data.iloc[-remainder:, :]
			if self.reduce_noise:
				data_in_window = self.get_top_k_power_data(data_in_window, self.k)
			transformed_features = self.get_features(data_in_window)
			final_data.append(transformed_features)

		return np.array(final_data)

	def make_columns(self):
		"""Generates column names for high-level features.

		Each data set are consisted of values of three axes

		Parameters
		----------
		data: panda data frame
			raw sensory data without the label

		Returns:
		----------
		np.array(final_data): ndarray
			transformed data for the entire data set
		"""

		columns = []
		features = ["mu", "std", "avg_resultant_acc", "bin", "mean_abs_difference"]
		for feat in features:
			if feat is "bin":
				# 10 bins for each axis, 30 bins in total for all 3 axes.
				for i in range(30):
					columns.append("bin_"+str(i)+"_"+self.sensor)
			elif feat is "avg_resultant_acc":
				columns.append("avg_resultant_acc_"+self.sensor)
			else:
				for axis in ["x", "y", "z"]:
					columns.append(feat+"_"+axis+"_"+self.sensor)
		return columns

	def get_top_k_power_data(self, data, k):
		""" Selects data with the top-k powers density.

		Parameters:
		----------
		data: pandas data frame
		 	raw data of one sliding window
		k: int
			the number of the top-k powers,
			only specified when reduce_noise is true (default is 0).

		Returns:
		----------
		DataFrame
			data with top-k powers density
		"""
	    columns = data.columns.values
	    n = data.shape[0]
	    final_data = []
	    
	    for col in columns:
	    	# transform data into Fourier coefficients 
	        data_fft = np.fft.fft(data[col])
	        psd = data_fft*np.conj(data_fft)/n
	        kth_psd = np.sort(psd.real)[-k]	
	        
	        # pick the data with top-k power density 
	        indices = psd > kth_psd
	        data_fft = data_fft*indices
	        # transform Fourier coefficients back into numberic data
	        data_ifft = np.fft.ifft(data_fft)
	        final_data.append(data_ifft.real)
	    
	    final_data = pd.DataFrame(np.array(final_data).T, columns=columns)
	    
	    return final_data

	def transform(self, X):
		""" Transform raw data

		Parameters
		----------
		X: panda data frame
			raw sensory data without the label

		Returns:
		----------
		funal_df: pandas data frame
			high-level features
		"""

		columns = self.make_columns()
		final_df = pd.DataFrame(data=[], columns=columns)
		#rows needed to calculate one sliding window amount of data
		#1s = 1000(ms); each row of data corresponds to 10(ms)

		# generate high-level features for all labels, 
		# but first, we need to get all the correct data for each class.
		for c in range(self.num_classes):
			start_idx = c*self.cut_off_number
			end_idx = (c+1)*self.cut_off_number
			transformed_data = self.generate_features(X.iloc[start_idx:end_idx, :])
			transformed_data = pd.DataFrame(data=transformed_data, columns=columns)
			final_df = final_df.append(transformed_data)

		return final_df


	def fit(self, X, num_classes):
		""" Build a feature generator from the raw data X.

		Parameters
		----------
		X: panda data frame
			raw sensory data without the label
		num_classes: int
			number of different classes

		Returns:
		----------
		"""
		self.num_classes = num_classes

		# we have to set the cut off number so that we know 
		# how many data points belong to each class
		self.cut_off_number = X.shape[0]//num_classes
		return self

	