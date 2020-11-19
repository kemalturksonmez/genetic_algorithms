# Authors:
# Kemal Turksonmez
# Arash Ajam
import numpy as np
import math
# This class contains methods dedicated to dataset analysis and mutation
class PD:

	# gets the mean and standard deviations for each column in the dataset
	# data - dataset
	# returns: 
	# mean - mean of each individual column
	# stdev - stdev of each individual column
	def mean_stdev(self, data):
		return np.mean(data, axis = 0), np.std(data, axis = 0)

	# standardize each value with z-score
	# data - data set
	# returns:
	# data - modified data set with standardized values
	def standardize_data(self, data):
		# get means and stdev
		mean, stdev = self.mean_stdev(data)
		# normalize each value
		for row in data:
			for i in range(len(row[:-1])):			
				if stdev[i] != 0:
					row[i] = (row[i] - mean[i])/stdev[i]
				else:
					row[i] = 0	
		return data

	# converts a feauture into a cyclic representation
	# num - index of value
	# total - total number of values
	# returns:
	# sin - y axis location of feature
	# cos - x axis location of feature
	def convert_to_cyclic(self, num, total):
		sin = math.sin(num*2*math.pi/total)
		cos = math.cos(num*2*math.pi/total)
		return sin, cos

	# converts each element in the data set to it's own array
	# data - data set
	# problemType - identifies if it's a classification/regression data set
	# returns:
	# tempDict/tempList - modified version of data set
	def convert_to_single_numpy(self, data, problemType):
		if problemType == "classification":
			tempDict = dict()
			# iterate through classes
			for key in data:
				tempDict[key] = []
				# iterate through rows in class dict
				for row in data[key]:
					tempRow = []
					# iterate through elements
					for ele in row:
						tempRow.append([ele])
					tempDict[key].append(np.array(tempRow))
			return tempDict
		else:
			tempList = []
			# iterate through rows in data set
			for row in data:
				tempRow = []
				# iterate through elements
				for ele in row:
					tempRow.append([ele])
				tempList.append(np.array(tempRow))
			return tempList

	# scans dataset to replace missing values with randomized values based on attribute range
	# dataset - observed dataset
	# attributes - attribute descriptions of a given dataset
	# returns:
	# returns modified dataset
	def replace_missing_val(self, dataset, attributes, replaceVal=None):
		for i in range(len(dataset)):
			for j in range(len(dataset[i])):
				# replace missing value with a random value based on min max range
				if dataset[i][j] == -1:
					# is real value
					if replaceVal:
						dataset[i][j] = replaceVal
					else:
						if attributes[j][0] == 1:
							dataset[i][j] = (attributes[j][1] - attributes[j][0]) * np.random.random_sample() + attributes[j][0]
						else:
							dataset[i][j] = int(((attributes[j][1] + 1) - attributes[j][0]) * np.random.random_sample() + attributes[j][0])
		return dataset

	# scans dataset to replace missing values with randomized values based on attribute range
	# dataset - observed dataset
	# attributes - attribute descriptions of a given dataset
	# returns:
	# returns modified dataset
	def remove_missing(self, dataset):
		tempData = []
		for i in range(len(dataset)):
			dont_remove = True
			for j in range(len(dataset[i])):	
				# replace missing value with a random value based on min max range
				if dataset[i][j] == -1:
					dont_remove = False
			if dont_remove:
				tempData.append(dataset[i])
		return tempData

	# removes the first column from the dataset
	# data - dataset
	# returns dataset with first column removed
	def remove_first_column(self, data):
		temp = []
		for i in range(len(data)):
			temp.append(data[i][1:len(data[0])])
		return temp

	# removes the second column from the dataset
	# data - dataset
	# returns dataset with last column removed
	def remove_last_column(self, data):
		temp = []
		deleted = []
		for i in range(len(data)):
			temp.append(data[i][0:-1])
			deleted.append(data[i][-1])
		return temp, deleted

	# shifts the first column to the end
	# data - dataset
	# returns dataset with first column shifted to the end
	def shift_first_column(self, data):
		temp = []
		for i in range(len(data)):
			temp.append(data[i][1:len(data)])
			temp[i] = np.append(temp[i],data[i][0:1])
		return temp

	# Find the min and max values for each column
	# data - dataset
	# containsClass - bool identifier if class is in dataset
	def minmax(self, data, containsClass):
		minmax = list()
		length = len(data[0])
		if containsClass:
			length -= 1
		for i in range(length):
			col_values = [row[i] for row in data]
			value_min = min(col_values)
			value_max = max(col_values)
			minmax.append([value_min, value_max])
		return minmax

	# Seperates data into lists by class number and removes id and class value column from dataset
	# dataset - observed dataset
	# attributes - attribute descriptions of a given dataset
	# returns:
	# dict that organized by class specific rows
	def split_by_class(self, dataset):
		separated = dict()
		for i in range(len(dataset)):
			row = dataset[i]
			class_value = row[-1]
			if (class_value not in separated):
				separated[class_value] = list()
			separated[class_value].append(row[0:len(row)])
		return separated
	
	
	# splits data into a test set for stratified cross validation
	# dataset - dataset
	# iteration - iteration of cross validation
	# returns: test and training data
	def stratifiedSplit(self, dataset, iteration):
		testData = []
		trainData = []
		for i in dataset:
			splitRange = int(len(dataset[i])/10)
			tempTest = dataset[i][splitRange * iteration:splitRange * (iteration+1)]
			if len(testData) == 0:
				testData = tempTest
			elif len(tempTest) > 0:
				testData = np.concatenate([testData, tempTest])
			temp1 = dataset[i][0:splitRange * iteration]
			temp2 = dataset[i][splitRange * (iteration+1):len(dataset[i])]
			if len(temp1) > 0 and len(temp2) > 0:
				if len(trainData) == 0:
					trainData = np.concatenate([temp1, temp2])
				else:
					trainData = np.concatenate([trainData, temp1, temp2])
			elif len(temp1) > 0:
				if len(trainData) == 0:
					trainData = temp1
				else:
					trainData = np.concatenate([trainData, temp1])
			else:
				if len(trainData) == 0:
					trainData = temp2
				else:		
					trainData = np.concatenate([trainData, temp2])
		return trainData, testData

	# splits data to pick 1/10 th of data out of every 10 data points
	# dataset - dataset
	# iteration - iteration of cross validation
	# returns: test and training data split for regression set
	def regressiveSplit(self, dataset, iteration):
		testData = []
		trainData = []
		for i in range(int(len(dataset)/10)):
			tempTest = dataset[10*i + iteration : 10*i + iteration + 1]
			if len(tempTest) > 0:
				if len(testData) > 0:
					testData = np.concatenate([testData, tempTest])
				else:
					testData = tempTest
			tempTrain1 = dataset[10*i : 10*i + iteration]
			tempTrain2 = dataset[10*i + iteration + 1: 10*i + 10]
			if len(tempTrain1) > 0:
				if len(trainData) > 0:
					trainData = np.concatenate([trainData, tempTrain1])
				else:
					trainData = tempTrain1 
			if len(tempTrain2) > 0:
				if len(trainData) > 0:
					trainData = np.concatenate([trainData, tempTrain2])
				else:
					trainData = tempTrain2 
		i += 1
		tempTrain = dataset[10*i : len(dataset)]
		if len(trainData) > 0:
			trainData = np.concatenate([trainData, tempTrain])
		return trainData, testData

	# Seperates data into lists by class number and removes id from dataset
	# dataset - observed dataset
	# attributes - attribute descriptions of a given dataset
	# returns:
	# dict that organized by class specific rows
	def split_by_class_1(self, dataset, attributes):
		separated = dict()
		for i in range(len(dataset)):
			row = dataset[i]
			class_value = row[-1]
			if (class_value not in separated):
				separated[class_value] = list()
			# range is for classes that contain an id at the start and a class number at the end
			# if row does not contain an id, the row will contain the first column
			start = 1
			if len(attributes) == (len(row) - 1):
				start = 0
			separated[class_value].append(row[start:])
		return separated

	# data - observed data
	# attributes - attribute descriptions of a given dataset
	# returns a dict with number of examples for each class in data
	def getDistributions(self, data, attributes):
		total = len(data)		
		seperated = self.split_by_class_1(data, attributes)
		dist_dict = {key: len(value)/total for key, value in seperated.items()}
				
		return dist_dict, seperated
	

