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
    """Trains the given data with the specified method.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :rtype: obj
    :retruns: trained sklearn classifier object
    """
    classifier = get_classifier(method)

    if classifier:
        classifier.fit(data, target)
    return classifier


def classify_data(classifier, data):
    """Classifies the given data with the specified method.

    :type classifier: obj
    :param method: Classification object from sklearn

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :rtype: np.array
    :returns: array of classification labels (n_samples,)
    """
    return classifier.predict(data)


def get_classifier(method):
    """Creates an sklearn classifier object.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :rtype: obj
    :retruns: instantiated sklearn classifier object
    """
    classifier = None

    if method == 'lr':
        classifier = LogisticRegression(
            random_state=42,
            C=1000)
    elif method == 'knn':
        classifier = KNeighborsClassifier()
    elif method == 'svm':
        classifier = svm.SVC(
            kernel='poly',
            C=1)
    elif method == 'nn':
        classifier = MLPClassifier(
            solver='lbfgs',
            alpha=.001,
            activation='tanh')
    return classifier


def perform_cross_validation(method, data, target, folds=10):
    """Performs cross validation for the given method and data.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :type folds: int
    :param folds: Number of folds to perform in cross validation

    :rtype: np.array
    :retruns: Accuracy of each fold (folds,)
    """
    classifier = get_classifier(method)
    accuracy = cross_validate(classifier, data, target, cv=folds, n_jobs=-1)
    return accuracy['test_score']


def get_confusion_matrix(method, data, target):
    """Generates a confusion matrix for the given method and data.

    :type method: str
    :param method: Name of classifier to use (lr, svm, knn, nn)

    :type data: np.array
    :param data: input data array of (n_samples, n_features)

    :type target: np.array
    :param target: Labels for the data input (n_samples,)

    :rtype: np.array
    :retruns: array representing the confusion matrix (n_labels, n_labels)
    """
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
