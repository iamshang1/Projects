import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys
import glob
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random

srng = RandomStreams()

input = np.load('X_train_zca.npy')   
labels = np.genfromtxt('../data/y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

convolutional_layers = 6
feature_maps = [3,80,80,160,160,320,320]
filter_shapes = [(3,3),(3,3),(3,3),(3,3),(3,3),(3,3)]
image_shapes = [(32,32),(32,32),(16,16),(16,16),(8,8),(8,8)]
feedforward_layers = 1
feedforward_nodes = [2000]
classes = 10

class convolutional_layer(object):
    def __init__(self, input, output_maps, input_maps, filter_height, filter_width, image_shape, maxpool=None):
        self.input = input
        self.w = theano.shared(self.ortho_weights(output_maps,input_maps,filter_height,filter_width),borrow=True)
        self.b = theano.shared(np.zeros((output_maps,), dtype=theano.config.floatX),borrow=True)
        self.conv_out = conv2d(input=self.input, filters=self.w, border_mode='half')
        if maxpool:
            self.conv_out = downsample.max_pool_2d(self.conv_out, ds=maxpool, ignore_border=True)
        self.output = T.nnet.elu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    def ortho_weights(self,chan_out,chan_in,filter_h,filter_w):
        bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
        W = np.random.random((chan_out, chan_in * filter_h * filter_w))
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u.reshape((chan_out, chan_in, filter_h, filter_w))
        else:
            W = v.reshape((chan_out, chan_in, filter_h, filter_w))
        return W.astype(theano.config.floatX)
    def get_params(self):
        return self.w,self.b

class feedforward_layer(object):
    def __init__(self,input,features,nodes):
        self.input = input
        self.bound = np.sqrt(1.5/(features+nodes))
        self.w = theano.shared(self.ortho_weights(features,nodes),borrow=True)
        self.b = theano.shared(np.zeros((nodes,), dtype=theano.config.floatX),borrow=True)
        self.output = T.nnet.sigmoid(-T.dot(self.input,self.w)-self.b)
    def ortho_weights(self,fan_in,fan_out):
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u
        else:
            W = v
        return W.astype(theano.config.floatX)
    def get_params(self):
        return self.w,self.b        

class neural_network(object):
    def __init__(self,convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes):
        self.input = T.tensor4()        
        self.convolutional_layers = []
        self.convolutional_layers.append(convolutional_layer(self.input,feature_maps[1],feature_maps[0],filter_shapes[0][0],filter_shapes[0][1],image_shapes[0]))
        for i in range(1,convolutional_layers):
            if i==2 or i==4:
                self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1],image_shapes[i],maxpool=(2,2)))
            else:
                self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1],image_shapes[i]))
        self.feedforward_layers = []
        self.feedforward_layers.append(feedforward_layer(self.convolutional_layers[-1].output.flatten(2),20480,feedforward_nodes[0]))
        for i in range(1,feedforward_layers):
            self.feedforward_layers.append(feedforward_layer(self.feedforward_layers[i-1].output,feedforward_nodes[i-1],feedforward_nodes[i]))
        self.output_layer = feedforward_layer(self.feedforward_layers[-1].output,feedforward_nodes[-1],classes)
        self.params = []
        for l in self.convolutional_layers + self.feedforward_layers:
            self.params.extend(l.get_params())
        self.params.extend(self.output_layer.get_params())
        self.target = T.matrix()
        self.output = self.output_layer.output
        self.cost = -self.target*T.log(self.output)-(1-self.target)*T.log(1-self.output)
        self.cost = self.cost.mean()
        self.updates = self.adam(self.cost, self.params)
        self.propogate = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.classify = theano.function([self.input],self.output,allow_input_downcast=True)
        
    def adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.01, e=1e-8):
        updates = []
        grads = T.grad(cost, params)
        self.i = theano.shared(np.float32(0.))
        i_t = self.i + 1.
        fix1 = 1. - (1. - b1)**i_t
        fix2 = 1. - (1. - b2)**i_t
        lr_t = lr * (T.sqrt(fix2) / fix1)
        for p, g in zip(params, grads):
            self.m = theano.shared(p.get_value() * 0.)
            self.v = theano.shared(p.get_value() * 0.)
            m_t = (b1 * g) + ((1. - b1) * self.m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * self.v)
            g_t = m_t / (T.sqrt(v_t) + e)
            p_t = p - (lr_t * g_t)
            updates.append((self.m, m_t))
            updates.append((self.v, v_t))
            updates.append((p, p_t))
        updates.append((self.i, i_t))
        return updates
        
    def train(self,X,y,batch_size=None):
        if batch_size:
            indices = np.random.permutation(X.shape[0])[:batch_size]
            crop1 = np.random.randint(-5,6)
            crop2 = np.random.randint(-5,6)
            X = X[indices,:,:,:]
            if crop1 > 0:
                X = np.concatenate((X[:,:,crop1:,:],np.zeros((batch_size,3,crop1,32))),axis=2)
            elif crop1 < 0:
                X = np.concatenate((np.zeros((batch_size,3,-crop1,32)),X[:,:,:crop1,:]),axis=2)
            if crop2 > 0:
                X = np.concatenate((X[:,:,:,crop2:],np.zeros((batch_size,3,32,crop2))),axis=3)
            elif crop2 < 0:
                X = np.concatenate((np.zeros((batch_size,3,32,-crop2)),X[:,:,:,:crop2]),axis=3)
            y = y[indices]
        y = np.concatenate((y,np.arange(10))) #make sure y includes all possible labels
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        target = target[:-10,:] #drop extra labels inserted at end
        if random.random() < .5:
            X = X[:,:,:,::-1]
        return self.propogate(X,target)
    
    def predict(self,X):
        prediction = self.classify(X)
        return np.argmax(prediction,axis=1)

print "building neural network"
nn = neural_network(convolutional_layers,feature_maps,filter_shapes,feedforward_layers,feedforward_nodes,classes)

batch_size = 100

for i in range(25000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i loss: %f \r" % (i+1, cost))
    sys.stdout.flush()
    if (i+1)%100 == 0:
        preds = []
        for j in range(0,X_test.shape[0],batch_size):
             preds.append(nn.predict(X_test[j:j+batch_size,:]))
        pred = np.concatenate(preds)
        error = 1-float(np.sum(pred==y_test))/len(pred)
        print "test error at iteration %i: %.4f" % (i+1,error)
