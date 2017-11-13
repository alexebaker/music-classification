from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

# for cross validation
from sklearn.model_selection import train_test_split

# for normalization
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def get_cross_validate_sets(data, target):
    train_test_split(data, target, test_size=.1)
    return


def train_data(method, data, target):
    classifier = None

    if method == 'lr':
        classifier = LogisticRegression(C=1000)
    elif method == 'knn':
        classifier = KNeighborsClassifier()
    elif method == 'svm':
        classifier = svm.SVC()
    elif method == 'nn':
        classifier = MLPClassifier()

    if classifier:
        classifier.fit(data, target)
    return classifier


def classify_data(classifier, data):
    return classifier.predict(data)



