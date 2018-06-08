__title__ = "Part 2: Manual Grid Search"
__author__ = "David Chen"
__copyright__ = "Copyright 2018 David Chen"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "David Chen"
__status__ = "Development"

import numpy as np
import pandas as pd
import itertools as it
import json

from keras import backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, fbeta_score, make_scorer

from trainDeepCNN import *
del main;

## Configurations:
devMode = False; #set to false when testing actual code
dirPath = "DL3 Dataset/";
out_path = 'DL3 Dataset/outputs/'; #includes intermediate outputs
devSetFrac = 0.30;
bSize = 50;
eps = 1 if devMode else 5; #use 1 for code testing

## Reading the train and test meta-data files:
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
    paramGrid = {
        'lr': [0.001, 0.01],
        'decay': [1e-5],
        'momentum': [0],
        'nesterov': [True]
    }; #test case
else:
    sampIndTrain = np.random.randint(low=0, high=X_norm.shape[0], size=6000);
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X_norm[sampIndTrain,:], y[sampIndTrain], test_size=devSetFrac);
    paramGrid = {
        'lr': [0.001, 0.01, 0.1],
        'decay':[1e-5, 1e-6, 0],
        'momentum': [0, 0.8, 0.9],
        'nesterov': [True, False]
    };

## Iterative training:
keys, values = zip( * paramGrid.items() );
flatGrid = [dict(zip(keys, v)) for v in it.product(*values)];
del keys, values, paramGrid, X_norm; #conserve memory & avoid confusion

resultsDict = {
    'lr': [],
    'decay':[],
    'momentum': [],
    'nesterov': [],
    'trainF1': [],
    'validationF1':[]
};

for param in flatGrid:
    print("Parameters are: ", param)
    resultsDict['lr'].append(param['lr']);
    resultsDict['decay'].append(param['decay']);
    resultsDict['momentum'].append(param['momentum']);
    resultsDict['nesterov'].append(param['nesterov']);

    model = setupModel(optimizer=SGD(lr=param['lr'], decay=param['decay'], momentum=param['momentum'], nesterov=param['nesterov']),
                       n_target=85, metrics=[customF1]);
    model.fit(Xtrain,
              ytrain,
              validation_data=(Xvalid, yvalid),
              epochs=eps,
              batch_size=bSize,
              verbose=1);

    ## Make prediction on training & dev-set fraction:
    pred = applyModel(model=model, newX=Xtrain, byBatch=True, batch_size=bSize);
    print( "Training-set F1 Score: %.2f" % f1_score(ytrain, pred, average='samples') )
    resultsDict['trainF1'].append( f1_score(ytrain, pred, average='samples') );

    pred = applyModel(model=model, newX=Xvalid, byBatch=True, batch_size=bSize);
    print( "Dev-set F1 Score: %.2f" % f1_score(yvalid, pred, average='samples') )
    resultsDict['validationF1'].append( f1_score(yvalid, pred, average='samples') );

    ## (IMPORTANT!) Delete model per iteration:
    del model, pred;

## Print & export:
print(resultsDict)
m = max(resultsDict['validationF1']);

print('Best validation F1 score: ' + str(m) + ', resulting from hyperparameters: ')
myIndex = resultsDict['validationF1'].index(m);
print('Learning rate: ', resultsDict['lr'][myIndex])
print('Decay: ', resultsDict['decay'][myIndex] )
print('Momentum: ', resultsDict['momentum'][myIndex])
print('nesterov', resultsDict['nesterov'][myIndex])

print("Saving grid-search record as JSON....")
with open('gridSearch.json', 'w') as fp:
    json.dump(resultsDict, fp);
