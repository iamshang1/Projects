'''
Convolutional variational autoencoder in Keras.

Based off dense variational autoencoder from
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
'''

import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Dropout
from keras.layers import Convolution2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K
from keras import objectives
import warnings

class conv_variational_autoencoder(object):
    '''
    variational autoencoder class
    
    parameters:
      - image_size: tuple
        height and width of images
      - channels: int
        number of channels in input images
      - conv_layers: int
        number of encoding/decoding convolutional layers
      - feature_maps: list of ints
        number of output feature maps for each convolutional layer
      - filter_shapes: list of tuples
        convolutional filter shape for each convolutional layer
      - strides: list of tuples
        convolutional stride for each convolutional layer
      - dense_layers: int
        number of encoding/decoding dense layers
      - dense_neurons: list of ints
        number of neurons for each dense layer
      - dense_dropouts: list of float
        fraction of neurons to drop in each dense layer (between 0 and 1)
      - latent_dim: int
        number of dimensions for latent embedding
      - activation: string (default='relu')
        activation function to use for layers
      - eps_mean: float (default = 0.0)
        mean to use for epsilon (target distribution for embedding)
      - eps_std: float (default = 1.0)
        standard dev to use for epsilon (target distribution for embedding)
       
    methods:
      - train(data,batch_size,epochs=1,history=False,checkpoint=False,
              filepath=None)
        train network on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
      - return_embeddings(data)
        return the embeddings for given data
      - generate(embedding)
        return a generated output given a latent embedding
    '''

    def __init__(self,image_size,channels,conv_layers,feature_maps,filter_shapes,
                 strides,dense_layers,dense_neurons,dense_dropouts,latent_dim,
                 activation='relu',eps_mean=0.0,eps_std=1.0):
        
        #check that arguments are proper length
        if len(filter_shapes)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of \
                             filter_shapes list")
        if len(strides)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of \
                             strides list")
        if len(feature_maps)!=conv_layers:
            raise Exception("number of convolutional layers must equal length of \
                             feature_maps list")
        if len(dense_neurons)!=dense_layers:
            raise Exception("number of dense layers must equal length of \
                             dense_neurons list")
        if len(dense_dropouts)!=dense_layers:
            raise Exception("number of dense layers must equal length of \
                             dense_dropouts list")
        
        #even shaped filters may cause problems in theano backend
        even_filters = [f for pair in filter_shapes for f in pair if f % 2 == 0]
        if K.image_dim_ordering() == 'th' and len(even_filters) > 0:
            warnings.warn('Even shaped filters may cause problems in Theano backend')
        
        self.eps_mean = eps_mean
        self.eps_std = eps_std
        self.image_size = image_size
        
        #define input layer
        if K.image_dim_ordering() == 'th':
            self.input = Input(shape=(channels,image_size[0],image_size[1]))
        else:
            self.input = Input(shape=(image_size[0],image_size[1],channels))
                    
        #define convolutional encoding layers
        self.encode_conv = []
        layer = Convolution2D(feature_maps[0],filter_shapes[0],padding='same',
                              activation=activation,strides=strides[0])(self.input)
        self.encode_conv.append(layer)
        for i in range(1,conv_layers):
            layer = Convolution2D(feature_maps[i],filter_shapes[i],
                                  padding='same',activation=activation,
                                  strides=strides[i])(self.encode_conv[i-1])
            self.encode_conv.append(layer)
        
        #define dense encoding layers
        self.flat = Flatten()(self.encode_conv[-1])
        self.encode_dense = []
        layer = Dense(dense_neurons[0],activation=activation)\
                (Dropout(dense_dropouts[0])(self.flat))
        self.encode_dense.append(layer)
        for i in range(1,dense_layers):
            layer = Dense(dense_neurons[i],activation=activation)\
                    (Dropout(dense_dropouts[i])(self.encode_dense[i-1]))
            self.encode_dense.append(layer)
        
        #define embedding layer
        self.z_mean = Dense(latent_dim)(self.encode_dense[-1])
        self.z_log_var = Dense(latent_dim)(self.encode_dense[-1]) 
        self.z = Lambda(self._sampling, output_shape=(latent_dim,))\
                 ([self.z_mean, self.z_log_var]) 
                
        #save all decoding layers for generation model
        self.all_decoding=[]
                
        #define dense decoding layers
        self.decode_dense = []
        layer = Dense(dense_neurons[-1], activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.z))
        for i in range(1,dense_layers):
            layer = Dense(dense_neurons[-i-1],activation=activation)
            self.all_decoding.append(layer)
            self.decode_dense.append(layer(self.decode_dense[i-1]))
        
        #dummy model to get image size after encoding convolutions
        self.decode_conv = []
        if K.image_dim_ordering() == 'th':
            dummy_input = np.ones((1,channels,image_size[0],image_size[1]))
        else:
            dummy_input = np.ones((1,image_size[0],image_size[1],channels))
        dummy = Model(self.input, self.encode_conv[-1])
        conv_size = dummy.predict(dummy_input).shape
        layer = Dense(conv_size[1]*conv_size[2]*conv_size[3],activation=activation)
        self.all_decoding.append(layer)
        self.decode_dense.append(layer(self.decode_dense[-1]))
        reshape = Reshape(conv_size[1:])
        self.all_decoding.append(reshape)
        self.decode_conv.append(reshape(self.decode_dense[-1]))
        
        #define deconvolutional decoding layers
        for i in range(1,conv_layers):
            if K.image_dim_ordering() == 'th':
                dummy_input = np.ones((1,channels,image_size[0],image_size[1]))
            else:
                dummy_input = np.ones((1,image_size[0],image_size[1],channels))
            dummy = Model(self.input, self.encode_conv[-i-1])
            conv_size = list(dummy.predict(dummy_input).shape)
            
            if K.image_dim_ordering() == 'th':
                conv_size[1] = feature_maps[-i]
            else:
                conv_size[3] = feature_maps[-i]
            
            layer = Conv2DTranspose(feature_maps[-i-1],filter_shapes[-i],
                                    padding='same',activation=activation,
                                    strides=strides[-i])
            self.all_decoding.append(layer)
            self.decode_conv.append(layer(self.decode_conv[i-1]))
        
        layer = Conv2DTranspose(channels,filter_shapes[0],padding='same',
                                activation='sigmoid',strides=strides[0])
        self.all_decoding.append(layer)
        self.output=layer(self.decode_conv[-1])

        #build model
        self.model = Model(self.input, self.output)
        self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=self.optimizer, loss=self._vae_loss)
        print "model summary:"
        self.model.summary()
        
        #model for embeddings
        self.embedder = Model(self.input, self.z_mean)
        
        #model for generation
        self.decoder_input = Input(shape=(latent_dim,))
        self.generation = []
        self.generation.append(self.all_decoding[0](self.decoder_input))
        for i in range(1, len(self.all_decoding)):
            self.generation.append(self.all_decoding[i](self.generation[i-1]))
        self.generator = Model(self.decoder_input, self.generation[-1])
        
    def _sampling(self,args):
        '''
        sampling function for embedding layer
        '''
        z_mean,z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean),mean=self.eps_mean,
                                  stddev=self.eps_std)
        return z_mean + K.exp(z_log_var) * epsilon
        
    def _vae_loss(self,input,output):
        '''
        loss function for variational autoencoder
        '''
        input_flat = K.flatten(input)
        output_flat = K.flatten(output)
        xent_loss = self.image_size[0] * self.image_size[1] \
                    * objectives.binary_crossentropy(input_flat,output_flat)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) 
                  - K.exp(self.z_log_var), axis=-1)
        return xent_loss + kl_loss
        
    class _LossHistory(Callback):
        '''
        loss history for training variational autoencoder
        '''
        def on_train_begin(self, logs={}):
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
        
    def train(self,data,batch_size,epochs=1,validation_data=None,history=False,
              checkpoint=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            input data
          - batch_size: int
            number of records per batch
          - epochs: int (default: 1)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation dataZ
          - history: boolean (default: False)
            whether or not to record training history
          - checkpoint: boolean (default: False)
            whether or not to save model after each epoch
          - filepath: string (optional)
            path to save model if checkpoint is set to True
        
        outputs:
            None
        '''
        if checkpoint==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        callbacks = []
        if history:
            self.history = self._LossHistory()
            callbacks.append(self.history)
        if checkpoint:
            callbacks.append(ModelCheckpoint(filepath))
        
        self.model.fit(data,data,batch_size,epochs=epochs,shuffle=True,
                       validation_data=(data,data),callbacks=callbacks)
    
    def save(self,filepath):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        self.model.save_weights(filepath)
        
    def load(self,filepath):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.model.load_weights(filepath)
    
    def return_embeddings(self,data):
        '''
        return the embeddings for given data
        
        parameters:
          - data: numpy array
            input data
        
        outputs:
            numpy array of embeddings for input data
        '''
        return self.embedder.predict(data)

    def generate(self,embedding):
        '''
        return a generated output given a latent embedding
        
        parameters:
          - data: numpy array
            latent embedding
        
        outputs:
            numpy array of generated output
        '''
        return self.generator.predict(embedding)

if __name__ == "__main__":

    import os
    from keras.datasets import mnist
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    #define parameters
    image_size = (28,28)
    channels = 1
    batch_size = 100
    conv_layers = 2
    feature_maps = [16,16]
    filter_shapes = [(3,3),(3,3)]
    strides = [(1,1),(2,2)]
    dense_layers = 1
    dense_neurons = [128]
    dense_dropouts = [0]
    latent_dim = 2
    epochs = 10
    
    #load data
    print "loading data"
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    #reshape to 4d tensors
    if K.image_dim_ordering() == 'th':
        tensor_shape = (1,image_size[0],image_size[1])
    else:
        tensor_shape = (image_size[0],image_size[1],1)
    x_train = x_train.reshape((x_train.shape[0],) + tensor_shape)
    x_test = x_test.reshape((x_test.shape[0],) + tensor_shape)
    
    #build autoencoder
    print "building variational autoencoder"
    autoencoder = conv_variational_autoencoder(
                  image_size,channels,conv_layers,feature_maps,filter_shapes,
                  strides,dense_layers,dense_neurons,dense_dropouts,latent_dim)

    #load saved weights if they exist
    if os.path.isfile("./test.dat"):
        autoencoder.load("./test.dat")
    else:
        autoencoder.train(x_train,batch_size,epochs=epochs,
                          validation_data=(x_test,x_test),
                          checkpoint=True,filepath="./test.dat")
    
    #return embeddings on test data
    x_test_encoded = autoencoder.return_embeddings(x_test)
    
    #plot latent space
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.savefig('latent_space.png')
    plt.show()
    
    #show generated samples
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = autoencoder.generate(z_sample)
            digit = x_decoded.reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig('generated_samples.png')
plt.show()
