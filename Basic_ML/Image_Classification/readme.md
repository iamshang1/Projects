# Image Classification

This exercise compares image classification accuracy using two methods:
 - "Traditional" computer vision features (e.g., haralick features and local binary patterns) are fed into a support vector machine classifier
 - A convolutional neural network is used to automatically detect features and classify images

This exercises utilizes the Python Mahotas package to generate computer vision features and the Python Theano package to create a convolutional
neural network. A dataset of 90 images is used. 30 images are of buildings, 30 images are of outdoor scenery, and 30 images are of text. 
The premise and data for this exercise were obtained from "Building Machine Learning Systems in Python" by Coelho and Richert.

##### Example building

![building](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Image_Classification/building.jpg)

##### Example scenery

![scenery](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Image_Classification/scene.jpg)

##### Example text

![text](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Image_Classification/text.jpg)

A support vector machine with a radial bias kernel is used for classification. The following features are used to train the support vector machine:
 - Round each pixel to the nearest 6-bit color (64 colors), then use the count of pixels for each color 
 - Haralick Features
 - Local Binary Patterns
 - Speeded Up Robust Features
 
The follow architecture is used for the convolutional neural network:
 - Image shrunk to 92x126 pixels and 3 feature maps (R,G,B)
 - Convolutional layer with 10x10 filter size and 5 feature maps 
 - Max pooling layer with 2x2 pooling size
 - Convolutional layer with 10x10 filter size and 5 feature maps 
 - Max pooling layer with 2x2 pooling size
 - Dense feedforward layer with 100 nodes
 
The dataset consists of 90 samples. 81 of these are used for the training set and 9 for the test set. Both the training and test set are stratified so that
each set is composed of an equal number of images from each class. Note that the same test set is used for all methods.

## Results

Test Set Accuracy of Support Vector Machine:

 - Using Only Color Percentage: 77.78
 - Using Only Haralick Features: 100.00
 - Using Only Local Binary Patterns: 88.89
 - Using Only Speeded Up Robust Features: 88.89
 - Using All Methods Combined: 88.89

Test Set Accuracy of Convolutional Neural Net: 100.00