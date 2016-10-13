##Comparing Deep Learning Training Methodologies

The vanishing gradient problem makes it difficult to train deep neural networks with many layers,
but recent advances in training methodologies have mitigated the issues caused by vanishing gradients.
This exercise compares the effectiveness of various deep learning training methodologies. The MNIST 
digits dataset is used as the train/test dataset for this exercise.

The following training methodologies are compared:

**Vanilla Feedforward Neural Network - 2 Layers**

 - Two hidden layers, 1000 sigmoid nodes each with orthogonal weight initialization
 - Final softmax layer with 10 nodes
 - 3000 training iterations using ADAM gradient descent and no regularization
 
**Vanilla Feedforward Neural Network - 4 Layers**

 - Four hidden layers, 1000 sigmoid nodes each with orthogonal weight initialization
 - Final softmax layer with 10 nodes
 - 3000 training iterations using ADAM gradient descent and no regularization
 
**Batch Normalized Neural Network - 4 Layers**

 - Four hidden layers, 1000 sigmoid nodes each with orthogonal weight initialization
 - Final softmax layer with 10 nodes
 - Batch Normalization used to normalize activation input for each hidden layer (normalization occurs after
 linear transform and before sigmoid activation)
 - 3000 training iterations using ADAM gradient descent and no regularization
 
**Residual Neural Network - 4 Layers**

 - Four hidden layers, 1000 sigmoid nodes each with orthogonal weight initialization
 - Final softmax layer with 10 nodes
 - Output of previous hidden layer(s) added to input of following hidden layer 
 - 3000 training iterations using ADAM gradient descent and no regularization
 
**Batch Normalized Residual Neural Network - 4 Layers**

 - Four hidden layers, 1000 sigmoid nodes each with orthogonal weight initialization
 - Final softmax layer with 10 nodes
 - Output of previous hidden layer(s) added to input of following hidden layer 
 - Batch Normalization used to normalize activation input for each hidden layer (normalization occurs after
linear transform and before sigmoid activation)
 - 3000 training iterations using ADAM gradient descent and no regularization
 
**Restricted Boltmann Machine Pre-training - 4 Layers**
 
 - Four hidden layers, 1000 sigmoid nodes each
 - Final softmax layer with 10 nodes
 - Weights in each hidden layer pre-trained using RBM with Contrastive Divergence
 - 3000 training iterations using ADAM gradient descent and no regularization

### Results

![test_accuracy_1](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Deep_Learning/test_accuracy_1.png)

![test_accuracy_2](https://github.com/iamshang1/Projects/blob/master/Advanced_ML/Deep_Learning/test_accuracy_2.png)

### Conclusions

Based on our results, batch normalized residual neural networks have the highest test set accuracy
of the various deep learning training methodologies.