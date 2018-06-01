#!/usr/bin/env python3
# Minimal test case
# The goal is to make sure the helper functions work...

__title__ = "Minimal Test Case"
__author__ = "David Chen"
__copyright__ = "Copyright 2018 David Chen"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Chen"
__status__ = "Development"

import numpy as np
from keras import backend as K

## Test Case 1:
print("Loading main functionality...")
def std_normalize(X, normAxis=0):
    '''
    Standard normalizes a feature matrix X with dimension (m, x, y, c). Adapted from the Keras Starter
    '''
    mean_img = X.mean(axis=normAxis);
    std_dev = X.std(axis=normAxis);
    X_norm = (X - mean_img) / std_dev;
    return X_norm;
mat =  np.ones((1, 227,227,3));
print( std_normalize(mat) )


## Test Case 2 (IMPORTANT!):
def customF1(y_true, y_pred):
    '''
    Keras-specific F1-Macro Score. Adapted from Paddy, StackOverflow
    The precision and recall child functions are batch-wise only
    :param y_true, y_pred: truth and prediction, respectively
    '''
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)));
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)));
        return true_positives / (predicted_positives + K.epsilon());
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)));
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)));
        return true_positives / (possible_positives + K.epsilon());
    precision = precision(y_true, y_pred);
    recall = recall(y_true, y_pred);
    return 2*((precision*recall)/(precision+recall+K.epsilon())); #debugged

print( customF1([1, 0, 1, 0], [1, 0, 1, 0]) )
print( customF1([1, 0, 1, 0], [1, 0, 1, 0]) )
print( customF1([1, 0, 1, 0], [0, 1, 0, 1]) )
