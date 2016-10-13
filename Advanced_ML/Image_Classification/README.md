# Image Classification

## Overview
This exercise applies image recognition techniques to classify images in the CIFAR-10 dataset.
CIFAR-10  is an established computer-vision dataset used for object recognition. It consists of 60,000 32x32 color
images containing one of 10 object classes, with 6000 images per class.

![cifar10](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Image_Classification/cifar-10.png)

## Preprocessing
The following preprocessing was applied to each image before sending to our classifier:
- 50% chance of horizontally flipping the image
- random horizontal and/or vertical translation by 5 pixels
- ZCA whitening

## Model Description
We used a convolutional neural network as our image classifier due to the proven effectiveness of these models
for image recognition tasks. We used the following architecture/features for our network: 
- 2 convolutional layers using 3x3 filter shape, 80 feature maps each, ELU activation
- maxpool with 2x2 window
- 2 convolutional layers using 3x3 filter shape, 160 feature maps each, ELU activation
- maxpool with 2x2 window
- 2 convolutional layers using 3x3 filter shape, 320 feature maps each, ELU activation
- feedforward layer of 2000 nodes, sigmoid activation
- softmax layer
- adam gradient descent
- orthogonal weight initialization

This model achieved 91% test accuracy on a test set of 10,000 images. It took approximately two hours to train
on an Nvidia GTX970 GPU.

## Instructions for Running Model
Ensure the following Python packages are installed on your instance:
- numpy
- sklearn
- theano
- scipy

Once your environment has been setup, download the project files and run the following:
- **generate input data:** python preprocessing.py *(Note that the dataset is downloaded from an AWS S3 archive which may no longer be available)*
- **check accuracy on cross validation:** python cross_validation_model.py
- **predict on test set:** python production_model.py

Predictions are saved every 5000 iterations to *prediction.txt*.