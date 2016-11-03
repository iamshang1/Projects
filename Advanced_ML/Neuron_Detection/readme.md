# Automated Neuron Detection

## Overview
Calcium imaging is a common technique used to capture images of neuron activity. 
Once these images are captured, researchers often have to go through the time-consuming
task of manually identifying where the neurons are located in each image.

![movie](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Neuron_Detection/movie.gif)
![zooming](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Neuron_Detection/zooming.gif)

This exercise attempts to automate the process of neuron detection in calcium images
using machine learning techniques. The dataset used for this exercise consists of 19 training
and 9 test sets of neuron images downloaded from [http://neurofinder.codeneuro.org/](http://neurofinder.codeneuro.org/).
We use a convolutional neural networks with sliding window as our model for detecting neurons.

## Preprocessing
Each set of images in the codeneuro dataset is composed of several thousand images
over time of the same group of neurons. For each set of images, we create a single image by 
taking the mean pixel intensity for each pixel over time and clipping any pixel intensities three
standard deviations above the mean (across all pixels).

Before feeding the images to the convolutional network, each image was put through a standard
scalar so pixel values were normalized with mean 0 and standard deviation 1. Furthermore, random
horizontal and vertical flipping was used on the images during training.

Most of the original images are 512x512 pixels in shape. All images were 0-padded to 552x552 pixels
before they were sent to the convolutional neural network.

## Convolutional Network Architecture
Our convolutional neural network uses a sliding window of 40x40 pixels to predict a probability
for each of the center 20x20 pixels in the window. This probability represents the likelihood
that the given pixel belongs to a neuron.

We used the following architecture for our convolutional network: 
- 40x40 pixel sliding window input
- 3 convolutional layers using 3x3 filter shape, 50 feature maps each, ELU activation
- maxpool with 2x2 window
- 3 convolutional layers using 3x3 filter shape, 100 feature maps each, ELU activation
- feedforward layer of 2000 nodes, tanh activation
- output layer of 400 nodes representing 20x20 pixels in center of window
- adam gradient descent
- orthogonal weight initialization
- batch size 100, 10000 training iterations

During training, we were also careful to ensure that each training batch was
evenly split between pixels belonging to neurons and pixels that did not belong to
neurons.

During testing, we ran the sliding window with a stride of 1 across the test image. This means
every pixel was predicted multiple times. For each pixel, we took the mean probability across
all predictions and rounded it to 1 or 0 to determine the final label for that pixel.

## Model Performance
The model achieved the following performance metrics across the 9 test datasets:

![score](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Neuron_Detection/score.png)

This is as good as or better than most of the traditional models which use PCA, ICA,
or NMF. For additional information about how these metrics are calculated, refer to the
[codeneuro website](http://neurofinder.codeneuro.org/).

The model takes approximately three hours to run on a single NVIDIA GTX 970 GPU.

## Instructions to Run Model
Ensure the following Python packages are installed on your instance:
- numpy
- sklearn
- skimage
- theano
- scipy
- matplotlib

The datasets used for this project can be downloaded from [http://neurofinder.codeneuro.org/](http://neurofinder.codeneuro.org/).
Once these datasets have been downloaded, run the following commands. Note that preprocessing.py must be run for each of the
datasets that you wish to train/test on.

- **play images as a movie:** python play_animation.py \<PATH_TO_NEUROFINDER_FOLDER\>
- **preprocess images:** python preprocessing.py \<PATH_TO_NEUROFINDER_FOLDER\> \<PATH_TO_OUTPUT_FOLDER\>
- **run conv net with cross validation:** python conv2d_crossvalidation.py \<PATH_TO_PREPROCESSED_TRAIN_DATA\>
- **use conv net to predict test labels:** python conv2d_predict.py \<PATH_TO_PREPROCESSED_TRAIN_DATA\> \<PATH_TO_PREPROCESSED_TEST_DATA\>

Predictions are saved after 10000 iterations to *submission.json*.