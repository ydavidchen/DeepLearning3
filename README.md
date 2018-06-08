[![Build Status](https://travis-ci.com/ydavidchen/DeepLearning3.svg?token=UysmqA8KAVt84WN1PXEP&branch=master)](https://travis-ci.com/ydavidchen/DeepLearning3)

# DeepLearning\#3 Coding Challenge

## Background

* Problem statement: Predict whether each of 85 attributes (with unknown identity) based on pixel values.
* Given the correlation structure across features, simple machine-learning models assuming conditional independence (e.g. Naive Bayes) would not be appropriate.
* Deep learning, particularly convolutional neural networks (CNN), has demonstrated robust performance in similar tasks.

## Approach

* Prior to modeling, both the training and test data have been extracted and reshaped as 227x227x3 numpy arrays. See script `processData.py`.
* The original implementation of ImageNet (AlexNet), later implemented in library Keras, was modified to accommodate the present multitask learning problem (there the number of tasks is 85) with up to 30 epochs (with _early stopping_):
  - Instead of "accuracy", a custom implementation of F1 Score was used as the metric for optimization for each epoch.
  - Stochastic gradient descent with 0.9 momentum was used to optimize the _binary_crossentropy_ objective.
  - A _sigmoid_ activation layer (instead of _softmax_ proposed by the original ImageNet) was implemented to estimate class labels. These are implemented in the script `trainDeepCNN.py`.
* Training performance:
  - On final epoch (no.26): `loss: 0.1113 - customF1: 0.9372 - val_loss: 0.3070 - val_customF1: 0.8368`
  - Training-set accuracy: 96.29%
  - Dev-set accuracy: 83.69%
  - Training-set F1 Score: 0.96
  - Dev-set F1 Score: 0.83

## File Structure

* `processData.py`: Python script to read images and and export `.npy` files for cloud/cluster ML.
* `trainDeepCNN.py`: Python script to train deep CNN and apply to test set. This is the "meat and bones" of the project.
* `DL3 Dataset`: Directory for data
  - Neither the raw image datasets (>4 and >2Gb for training and test) nor the normalized $(128, 128, 3)$-dimensional `.npy` objects were not uploaded to GitHub due to size limit
  - The `.npy` objects were used for cluster / cloud computing.
* `Train_Deep_CNN.o142829`: Text file record for training output on Linux cluster.

## Implementation

### System

Data processing and normalization should be done as follows:

`python3 processData.py`

A Linux cluster (thanks to Dartmouth Research Computing) or cloud (thanks to GCP free trial) were used for training. Shell commands for `conda` virtual environment setup:

```
module load python/3.6-Miniconda
conda create --name DeepLearning3 python=3.6.1 #Proceed ([y]/n)? y ...
source activate DeepLearning3

conda install -c anaconda mkl
conda install mkl-service
conda install pandas
conda install -c conda-forge keras
```

#### Model Architecture

High Performance Computing (HPC, courtesy of Dartmouth) on the Linux cluster was used. Prior to actual execution, both Linux HPC and Google Cloud Platform (GCP) were also used for testing purposes.

The snippet below shows the latest model for submission:

```
python3 trainDeepCNN.py

# ...
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 227, 227, 3)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 55, 55, 96)        34944     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 27, 27, 96)        0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 27, 27, 96)        384       
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 31, 31, 96)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 27, 27, 256)       614656    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 256)       0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 13, 13, 256)       1024      
_________________________________________________________________
zero_padding2d_2 (ZeroPaddin (None, 15, 15, 256)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 384)       885120    
_________________________________________________________________
zero_padding2d_3 (ZeroPaddin (None, 15, 15, 384)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 13, 13, 384)       1327488   
_________________________________________________________________
zero_padding2d_4 (ZeroPaddin (None, 15, 15, 384)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 13, 13, 256)       884992    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 9216)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              37752832  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dropout_2 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 85)                348245    
=================================================================
Total params: 58,630,997
Trainable params: 58,630,293
Non-trainable params: 704

# ...
# source deactivate DeepLearning3
# conda env remove --name DeepLearning3 #Proceed ([y]/n)? y
```

The optimal hyperparameters for stochastic gradient descent (SGD) optimizers were determined by an iterative grid search on a subset of images using the same `customF1` scoring metric. All other hyperparameters were used at their default or according to the original publication (Krezhevsky et al. 2012). With 4x8=32 CPUs, each _epoch_ at batch size of 25 takes approximately 19 minutes to run.

## Limitations

* Model interpretability and difficulty in grid search
* System discrepancies (e.g. python and package versions)

## References

* Krizhevsky A et al. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems 25 (NIPS 2012).
  - Implementation by Shikar V. and Dan D. (GitHub)
* Chollet F et al. Keras <https://keras.io> (2015)

Copyright &copy; 2018 David Chen
