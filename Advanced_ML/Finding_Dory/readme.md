# Finding Dory

This exercise attempts to locate the character Dory in scenes taken from Pixar's Finding Nemo movie. A convolutional neural
network is trained to identify the character Dory. 150 snapshots of Dory and 150 snapshots without Dory from the movie are used
as training data for the convolutional neural net. Once the convolutional neural network has been trained, it is used to scan
the selected scenes using squares of various sizes. The square with the highest probability of containing Dory is then chosen
as the final location of Dory in the scene.

The follow architecture is used for the convolutional neural network:
 - Image image resized to 200x200 pixels and 4 feature maps (R,G,B,A)
 - Convolutional layer with 5x5 filter size and 5 feature maps 
 - Max pooling layer with 2x2 pooling size
 - Convolutional layer with 5x5 filter size and 5 feature maps 
 - Max pooling layer with 2x2 pooling size
 - Dense feedforward layer with 100 nodes

## Results

![scene1](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Finding_Dory/scene1.jpg)

![scene2](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Finding_Dory/scene2.jpg)

![scene3](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Finding_Dory/scene3.jpg)

![scene4](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Finding_Dory/scene4.jpg)

![scene5](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Finding_Dory/scene5.jpg)