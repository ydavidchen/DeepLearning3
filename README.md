# DeepLearning\#3 Coding Challenge

## Background

* Problem statement: to predict whether each of $85$ attributes (with unknown identity) based on pixel values
* Given the correlation structure across features, simple machine-learning models assuming conditional independence (e.g. Naive Bayes) would not be appropriate
* Deep learning, particularly convolutional neural networks (CNN), has demonstrated robust performance in such problems

## Approach

* Prior to modeling, both the training and test data have been extracted into 227x227x3 numpy arrays using the script `processData.py`.
* The original implementation of ImageNet (AlexNet), later implemented in library Keras, was modified to accommodate multi-task learning problem (there the number of tasks is 85) with 25 epochs.
  - Instead of "accuracy", a custom implementation of F1 Score was used as the metric for optimization for each epoch.
  - Stochastic gradient descent with 0.9 momentum was used to optimize the _binary_crossentropy_ objective.
  - A sigmoid final activation layer (instead of softmax proposed by the original ImageNet) was implemented to estimate class labels. These are implemented in the script `trainDeepCNN.py`.
* Training performance:
  - On final epoch: `loss: 0.3010 - customF1: 0.8104 - val_loss: 0.3588 - val_customF1: 0.7728`
  - Training Accuracy: 82.65%
  - Validation Accuracy: 77.27%
  - Training F1 score: 0.83
  - Dev-Set F1 score: 0.77

## File Structure

* `processData.py`: Python script to read images and and export `.npy` files for cloud/cluster ML.
* `trainDeepCNN.py`: Python script to train deep CNN and apply to test set. This is the "meat and bones" of the project.
* `DL3 Dataset`: Directory for data
  - Neither the raw image datasets (>4 and >2Gb for training and test) nor the normalized $(128, 128, 3)$-dimensional `.npy` objects were not uploaded to GitHub due to size limit
  - The `.npy` objects were used for cluster / cloud computing.
* `Train_Deep_CNN.o141174`: Text file record for training output on Linux cluster.

## Implementation

### System

A local MacOS was used for data processing and normalization as follows:

`python processData.py`

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

```
DEFAULT_USER$ python3 trainDeepCNN.py
#...
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

With 4x8 = 32 Linux-cluster CPUs, each _epoch_ takes approximately 18 minutes to run depending on the resource used, traffic (if cluster), etc.

## Credit

**Acknowledgements**

* Resources: Dartmouth Research Computing (Linux Cluster)
* Organizer: HackerEarth

**References**

* Chollet F et al. Keras <https://keras.io> (2015)
* Krizhevsky A et al. ImageNet Classification with Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems 25 (NIPS 2012)

Copyright &copy 2018 David Chen (will be transferred if winning or retained otherwise)
