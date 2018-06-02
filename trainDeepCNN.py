#!/usr/bin/env python3
# trainDeepCNN.py
# This script is to be run on the Cloud (GCP, AWS, etc) or Linux cluster

__title__ = "Part 2: Training the Deep Neural Network"
__author__ = "David Chen"
__copyright__ = "Copyright 2018 David Chen"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Chen"
__status__ = "Development"

from keras import backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

def main():
    ## Configurations:
    devMode = False; #set to false when testing actual code
    dirPath = "DL3 Dataset/";
    out_path = 'DL3 Dataset/outputs/'; #includes intermediate outputs
    devSetFrac = 0.10;
    bSize = 25;
    eps = 1 if devMode else 30; #use 1 for code testing

    ## Phase I: Data Loading & Training with Saving:
    ## Reading the train and test meta-data files:
    print("==================Phase I begins: CNN Model Training==================")
    train = pd.read_csv(dirPath+'train.csv');
    X_norm = np.load(out_path+'normalized_train_features.npy');

    ## Extract & set aside class labels:
    label_cols = list(set(train.columns) - set(['Image_name']));
    label_cols.sort();
    y = train[label_cols].values;

    ## Training-test split:
    if devMode:
        sampIndTrain = np.random.randint(low=0, high=X_norm.shape[0], size=500);
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_norm[sampIndTrain,:], y[sampIndTrain], test_size=devSetFrac);
    else:
        Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_norm, y, test_size=devSetFrac);
    del X_norm; #save memory & avoid confusion

    ## Set up & train the model:
    print("Setting up and computing on the model...")
    model = setupModel(optimizer=SGD(momentum=0.9, nesterov=True), n_target=85, metrics=[customF1]);
    print( model.summary() )
    callBackList = [EarlyStopping(monitor='val_loss', patience=6, mode='min'),
                    ModelCheckpoint(filepath='weights.best.eda.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)];
    # callBackList = [ModelCheckpoint(filepath='weights.best.eda.hdf5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)];
    model.fit(Xtrain,
              ytrain,
              validation_data=(Xvalid, yvalid),
              epochs=eps,
              batch_size=bSize,
              callbacks=callBackList,
              verbose=1);

    ## Save a copy of the model architecture as JSON for backup:
    print("Exporting a copy of the model as JSON as backup...")
    model_json = model.to_json();
    with open(out_path+"model.json", "w") as json_file:
        json_file.write(model_json);
    del model_json;

    ## Accuracy (which could be misleading in binary case!):
    scores = model.evaluate(Xtrain, ytrain, verbose=0);
    print("Training-set accuracy: %.2f%%" % (scores[1]*100))
    scores = model.evaluate(Xvalid, yvalid, verbose=0);
    print("Dev-set accuracy: %.2f%%" % (scores[1]*100))
    del scores;

    ## Make prediction on training & dev-set fraction:
    pred = applyModel(model=model, newX=Xtrain);
    print("Training-set F1 Score: %.2f" % f1_score(ytrain, pred, average='samples') )
    pred = applyModel(model=model, newX=Xvalid);
    print("Dev-set F1 Score: %.2f" % f1_score(yvalid, pred, average='samples') )
    del pred;

    ## Phase II: Application
    print("==================Phase II begins: Apply trained model to test set==================")
    test = pd.read_csv(dirPath+'test.csv');
    X_norm_test = np.load(out_path+'normalized_test_features.npy');

    if devMode:
        sampIndTest = np.random.randint(0, X_norm_test.shape[0], 100);
        X_norm_test = X_norm_test[sampIndTest, :];

    print("Making predictions about unknown test data:")
    model.load_weights('weights.best.eda.hdf5');
    myPredictions = applyModel(model=model, newX=X_norm_test);

    ## Generate submission file:
    print("==================Creating submission CSV file==================")
    subm = pd.DataFrame();
    subm['Image_name'] = test.Image_name;
    label_df = pd.DataFrame(data=myPredictions, columns=label_cols);
    subm = pd.concat([subm, label_df], axis=1);
    subm.to_csv(out_path+'submit_DavidChen.csv', index=False);
    print("Submission file created!")

def setupModel(optimizer, metrics, n_target, input_shape=(227,227,3), DROPOUT=0.5, n_category=2, loss='binary_crossentropy'):
    '''
    Function to set up an ImageNet Keras model for fitting a training data set
    Adapted for multi-target prediction
    Originally developed by Alex K et al., and implemented by Shikhar V. and Dan D.
    :param optimizer: Keras optimizer
    :param metrics: List of metrics. Either "accuracy" or custom function
    :param n_target: number of target variables
    :param input_shape: Here, it needs to be (x=227,y=227,c=3)
    :param DROPOUT: Proportion of nodes to dropout (regularization)
    :param n_category: number of classes per target variables
    :param loss: Keras' loss function of choice
    '''
    ## Initialize:
    model_input = Input(shape=input_shape)

    ## 1st conv layer (96x11x11):
    z = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation="relu")(model_input)
    z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
    z = BatchNormalization()(z)

    ## 2nd conv layer (256x5x5):
    z = ZeroPadding2D(padding=(2,2))(z)
    z = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation="relu")(z)
    z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
    z = BatchNormalization()(z)

    ## 3rd-5th conv layers (variable dimensions; see ImageNet paper):
    z = ZeroPadding2D(padding=(1,1))(z)
    z = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu")(z)

    z = ZeroPadding2D(padding=(1,1))(z)
    z = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation="relu")(z)

    z = ZeroPadding2D(padding=(1,1))(z)
    z = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation="relu")(z)

    z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(z)
    z = Flatten()(z)

    ## 2 dense layers:
    z = Dense(4096, activation="relu")(z)
    z = Dropout(DROPOUT)(z)

    z = Dense(4096, activation="relu")(z)
    z = Dropout(DROPOUT)(z)

    ## Finalize model & adjust dimensions:
    lastDimension = 1*n_target if n_category==2 else n_category*n_target;
    lastActivation = "sigmoid" if n_category==2 else "softmax";
    model_output = Dense(lastDimension, activation=lastActivation)(z);
    model = Model(model_input, model_output);

    ## Choose optimizer:
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics);
    return model;

def applyModel(model, newX):
    '''
    Make predictions on new data
    :param model: Keras model with architecture & optimizer
    :param newX: new data without target variable, which must be a numpy array with dim=(n,x,y,c)
    '''
    myPredictions = model.predict(newX).round().astype(np.int);
    return myPredictions;

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


if __name__ == '__main__':
    main();
