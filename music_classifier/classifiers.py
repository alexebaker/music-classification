from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

# for cross validation
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix

# for normalization
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm


def train_data(method, data, target):
    classifier = get_classifier(method)

    if classifier:
        classifier.fit(data, target)
    return classifier


def classify_data(classifier, data):
    return classifier.predict(data)


def get_classifier(method):
    classifier = None

    if method == 'lr':
        classifier = LogisticRegression(
            random_state=42,
            #solver='newton-cg',
            #penalty='l2',
            #multi_class='multinomial',
            C=1000)
    elif method == 'knn':
        classifier = KNeighborsClassifier()
    elif method == 'svm':
        classifier = svm.SVC(
            kernel='poly',
            C=0.1)
    elif method == 'nn':
        classifier = MLPClassifier(
            solver = 'lbfgs',
            alpha = .001,
            activation = 'tanh'
)

    return classifier


def perform_cross_validation(method, data, target):
    classifier = get_classifier(method)
    accuracy = cross_validate(classifier, data, target, cv=10, n_jobs=-1)
    return accuracy['test_score']


def get_confusion_matrix(method, data, target):
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        data,
        target,
        test_size=0.1,
        train_size=0.9,
        random_state=42)

    classifier = train_data(method, training_data, training_labels)
    classification = classify_data(classifier, testing_data)
    confusion = confusion_matrix(testing_labels, classification)
    return confusion
