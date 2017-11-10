"""from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division"""

import numpy as np

# for cross validation
from sklearn.model_selection import train_test_split

# for normalization
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def cross_validate_log_reg(data, target):
    train_test_split(data, target, test_size=.1, random_state=0)


def log_reg_train(data, target):
    train_data = data_normalize(data)

    lr = LogisticRegression(C=1000, random_state=0)
    lr.fit(train_data, target)
    return lr


def log_reg_classify(lr, data):
    test_data = data_normalize(data)

    return lr.predict(test_data)


def knn_train(data, target):
    train_data = data_normalize(data)

    knn = KNeighborsClassifier()
    knn.fit(train_data, target)
    return knn


def knn_classify(knn, data):
    test_data = data_normalize(data)

    return knn.predict(test_data)

def svm_train(data, target):
    train_data = data_normalize(data)

    svm_new = svm.SVC()
    svm_new.fit(train_data, target)
    return svm_new


def svm_classify(svm_new, data):
    test_data = data_normalize(data)

    return svm_new.predict(test_data)


def data_normalize(data):
    #norm = StandardScaler()
    norm = Normalizer()
    return norm.fit_transform(data)
