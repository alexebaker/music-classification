"""from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division"""

import numpy as np
#for cross validation
from sklearn.modelselection import train_test_split
#for normalization
from sklearn.preprocessing import StandardScalar
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def cross_validate_log_reg(data, target):
	train_test_split(data, target, test_size=.1, random_state=0)

def log_reg_train(data, target):
	train_data= data_normalize(data)

	lr= LogisticRegression(C=1000, random_state=0)
	lr.fit(train_data, target)
	return lr

def log_reg_classify(lr, data):
	test_data= data_normalize(data)

	return lr.predict(test_data)

def knn_train(data, target):
	train_data= data_normalize(data)

	knn= KNeighborsClassifier()
	knn.fit(train_data, target)
	return knn

def knn_classify(knn, data):
	test_data= data_normalize(data)

	return knn.predict(test_data)

def data_normalize(data):
	sc = StandardScalar()
	sc.fit(data)
	return sc.transform(data)