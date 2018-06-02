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

print("Loading main functionality...")
def std_normalize(X, normAxis=0):
    '''
    Standard normalizes a feature matrix X with dimension (m, x, y, c). Adapted from the Keras Starter
    '''
    mean_img = X.mean(axis=normAxis);
    std_dev = X.std(axis=normAxis);
    X_norm = (X - mean_img) / std_dev;
    return X_norm;

mat = 2.5*np.random.randn(5,227,227,3) + 3;
print( std_normalize(mat) )
