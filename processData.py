#!/usr/bin/env python
# processData.py 
# This script was run on my local machine where training/test images were downloaded
# The pixel numpy arrays were exported for cloud computing

__title__ = "Part 1: Converting and Exporting Pixel Matrices"
__author__ = "David Chen"
__copyright__ = "Copyright 2018 David Chen"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Chen"
__status__ = "Production"

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def main():
    TRAIN_PATH = 'DL3 Dataset/train_img/';
    TEST_PATH = 'DL3 Dataset/test_img/';
    OUT_PATH = 'DL3 Dataset/outputs/';

    ## Load, process & export training set:
    train = pd.read_csv('DL3 Dataset/train.csv');
    print ('Training data consists of {} images with {} attributes'.format(train.shape[0], train.shape[1]-1))

    train_img = [];
    for img_path in tqdm(train.Image_name.values):
        train_img.append(read_img(TRAIN_PATH + img_path));
    X_train = np.array(train_img, np.float32) / 255.;
    X_norm = std_normalize(X_train);
    del X_train;

    print('Dimension of training set: ')
    print(X_norm.shape)
    np.save(OUT_PATH+'normalized_train_features.npy', X_norm);
    print('Normalized training set saved!')

    ## Load, process & export test set:
    test = pd.read_csv('DL3 Dataset/test.csv');
    print ('Test data consists of {} images.'.format(test.shape[0]))

    test_img = [];
    for img_path in tqdm(test.Image_name.values):
        test_img.append(read_img(TEST_PATH + img_path));
    X_test = np.array(test_img, np.float32) / 255.;
    X_norm_test = std_normalize(X_test);
    del X_test;

    print('Dimension of test set (without labels): ')
    print(X_norm_test.shape)
    np.save(OUT_PATH+'normalized_test_features.npy', X_norm_test);
    print('Normalized test set saved!')

def read_img(img_path, x=227, y=227):
    '''
    Reads an image, resizes to x*y dimensions, and returns that image. Adapted from the Keras Starter
    '''
    img = cv2.imread(img_path)
    img = cv2.resize(img, (x,y))
    return img

def std_normalize(X, normAxis=0):
    '''
    Standard normalizes a feature matrix X with dimension (m, x, y, c). Adapted from the Keras Starter
    '''
    mean_img = X.mean(axis=normAxis);
    std_dev = X.std(axis=normAxis);
    X_norm = (X - mean_img) / std_dev;
    return X_norm;


if __name__ == '__main__':
    main()
