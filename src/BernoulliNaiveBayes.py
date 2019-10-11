###########################################################
# File name: BernoulliNaiveBayes.py
# Author: Thomas Racine
# Purpose: implement the Bernoulli Naive Bayes classification model
###########################################################

import numpy as np

# The function signatures for fit and predict are made to match scikit learn's function signatures to help implementation

# The model assumes that all the data has already been preprocess to be binary features (0 or 1)


class BernoulliNaiveBayes:

    def __init__(self):
        self.classes = np.array()  # list of the different classes, shape[n_classes]
        self.thetaK = np.array()  # list of marginal probabilities of the classes, shape[n_classes]
        self.thetaJK = np.array()  # Matrix with the probabilities P(x == 1 | y == k), shape[n_classes, n_features]


    '''
    The fit method where we learn our parameters
    Input:
        - X: The training vector set, shape[n_trainingSamples, n_features]
        - y: The target values, shape
    Output:
        none
    '''
    def fit(self, X, y):
        # we create an array with the different classes and an array with their counts
        self.classes, counts = np.unique(y, return_counts=True)
        # we compute the marginal probabilities P(y == k) using the size of y and the counts we got earlier
        self.thetaK = counts / y.shape[0]

        # we compute the conditional probabilities
        self.thetaJK = np.empty((self.classes.shape[0], X.shape[1]))
        # TODO: find a way to use vector operations instead of for loops if possible and time permits
        for k in range(self.classes.shape[0]):
            for j in range(X.shape[1]):
                examples = np.where(y == self.classes[k])  # we count the number of times we have a certain class
                instances = np.where(X[examples, j] == 1)  # we count the numbers of times x_j == 1 in the examples
                self.thetaJK[k, j] = instances + 1 / examples + 2  # we compute the conditional probability with Laplace Smoothing

    '''
    The predict function where we classify data based on the learned parameters
    Input:
        - X: The sample vector set, shape[n_samples, n_features]
    Output:
        - predictions: The prediction vector, shape[n_samples]

    '''
    def predict(self, X):
        predictions = np.array()
        classProb = np.array()
        for x in X:
            # we compute the log likelyhood of each class of the sample
            for i in self.classes.shape[0]:
                featureLikelyhood = 0
                # we compute the log Likely hood of the features for a given class
                for j in range(x.shape[0]):
                    featureLikelyhood += x[j] * np.log(self.thetaJK[i, j]) + (1 - x[j]) * np.log(1 - self.thetaJK[i, j])
                classProb.append(np.log(self.thetaK[i]) + featureLikelyhood)
            # we predict the class with the highest likelyhood
            predictions.append(self.classes[np.argmax(classProb)])
        return predictions

