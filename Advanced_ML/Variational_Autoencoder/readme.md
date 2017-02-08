#Convolutional Variational Autoencoders
Traiditional autoencoders take a high dimensional input and try to encode that
information into a lower-dimensional embedding with miniminal information loss. This
is achieved using a symmetrical funnel-shaped neural network -- the input and output layers have 
the same number of neurons, while layers closer to the middle of the network have
fewer neurons. The middle layer of the neural network represents the lower-dimension embedding for
the input data. When training the autoencoder, the goal of the network is to minimize the difference between
the output and the input across a wide variety of training examples.

In traditional autoencoders, the embeddings for input data can be sparsely distributed. 
Variational autoencoders expand on traditional autoencoders by forcing the embeddings created
from the training examples to fit a normal distribution. Because the embeddings are no
longer as sparse, the latent space between embeddings of real training examples can then
be used to generate new artificial examples.

This exercise creates convolutional variational autoencoder that can be used on
images and other 2d representations of data. We ran our autoencoder on the MNIST dataset and
embedded the original 784-dimension representation of each digit into a 2-dimensional embedding space.
The latent embedding space is then used to generate synthetic examples of digits.

##Results
The following image represents the 2-D latent space of the MNIST dataset. Each digit is colored
differently.

![latent_space](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Variational_Autoencoder/latent_space.png)

The following image represents synthetic digits generated from the latent space between real samples.

![generated_samples](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Variational_Autoencoder/generated_samples.png)