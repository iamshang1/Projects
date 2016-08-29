import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
import glob
from PIL import Image, ImageDraw
try:
    import cPickle as pickle
except:
    import pickle
import sys

sys.setrecursionlimit(10000)

regularization = 0.01
convolutional_layers = 2
feature_maps = [4,5,5]
filter_shapes = [(5,5),(5,5)]
poolsize = [(2,2),(2,2)]
feedforward_layers = 1
feedforward_nodes = [100]
classes = 2

class convolutional_layer(object):
    def __init__(self, input, output_maps, input_maps, filter_height, filter_width, poolsize=(2,2)):
        self.input = input
        self.bound = np.sqrt(6./(input_maps*filter_height*filter_width + output_maps*filter_height*filter_width//np.prod(poolsize)))
        self.w = theano.shared(np.asarray(np.random.uniform(low=-self.bound,high=self.bound,size=(output_maps, input_maps, filter_height, filter_width)),dtype=input.dtype))
        self.b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(output_maps)),dtype=input.dtype))
        self.conv_out = conv2d(input=self.input, filters=self.w)
        self.pooled_out = downsample.max_pool_2d(self.conv_out, ds=poolsize, ignore_border=True)
        self.output = T.tanh(self.pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    def get_params(self):
        return self.w,self.b

class feedforward_layer(object):
    def __init__(self,input,features,nodes):
        self.input = input
        self.bound = np.sqrt(1.5/(features+nodes))
        self.w = theano.shared(np.asarray(np.random.uniform(low=-self.bound,high=self.bound,size=(features,nodes)),dtype=theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(nodes)),dtype=theano.config.floatX))
        self.output = T.nnet.sigmoid(-T.dot(self.input,self.w)-self.b)
    def get_params(self):
        return self.w,self.b

class neural_network(object):
    def __init__(self,convolutional_layers,feature_maps,filter_shapes,poolsize,feedforward_layers,feedforward_nodes,classes,regularization):
        self.input = T.tensor4()
        self.convolutional_layers = []
        self.convolutional_layers.append(convolutional_layer(self.input,feature_maps[1],feature_maps[0],filter_shapes[0][0],filter_shapes[0][1],poolsize[0]))
        for i in range(1,convolutional_layers):
            self.convolutional_layers.append(convolutional_layer(self.convolutional_layers[i-1].output,feature_maps[i+1],feature_maps[i],filter_shapes[i][0],filter_shapes[i][1],poolsize[i]))
        self.feedforward_layers = []
        self.feedforward_layers.append(feedforward_layer(self.convolutional_layers[-1].output.flatten(2),flattened,feedforward_nodes[0]))
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
        for i in range(convolutional_layers+feedforward_layers+1):
            self.cost += regularization*(self.params[2*i]**2).mean()
        self.updates = self.adam(self.cost,self.params)
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
            X = X[indices,:,:,:]
            y = y[indices]
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        return self.propogate(X,target)
    def predict(self,X):
        prediction = self.classify(X)
        return (prediction, np.argmax(prediction,axis=1))
        
try:
    with open("conv_net.dat","rb") as pickle_nn:
        nn = pickle.load(pickle_nn)
    print "saved convoluation neural network loaded"        
except:
    print "loading images"
    X = np.empty((0,4,200,200))
    y = []

    dory_path = glob.glob('Dory/*.PNG')
    dory = [Image.open(img) for img in dory_path]
    dory = [img.resize((200,200), Image.ANTIALIAS) for img in dory]
    for image in dory:
        img = np.array(image,dtype='float64')/256
        img = img.transpose(2, 0, 1).reshape(1,4,200,200)
        X = np.concatenate((X,img),axis=0)
        y.append(1)

    not_dory_path = glob.glob('Not_Dory/*.PNG')
    not_dory = [Image.open(img) for img in not_dory_path]
    not_dory = [img.resize((200,200), Image.ANTIALIAS) for img in not_dory]
    for image in not_dory:
        img = np.array(image,dtype='float64')/256
        img = img.transpose(2, 0, 1).reshape(1,4,200,200)
        X = np.concatenate((X,img),axis=0)
        y.append(0)
        
    y = np.array(y)
    
    flattened = list(X.shape[2:])
    for i in range(convolutional_layers):
        flattened[0] = flattened[0] - filter_shapes[i][0] + 1
        flattened[1] = flattened[1] - filter_shapes[i][1] + 1
        flattened[0] = flattened[0]/poolsize[i][0]
        flattened[1] = flattened[1]/poolsize[i][1]
    flattened = np.prod(flattened)
    flattened *= feature_maps[-1]
    
    print "building convolutional neural network"
    nn = neural_network(convolutional_layers,feature_maps,filter_shapes,poolsize,feedforward_layers,feedforward_nodes,classes,regularization)
    for i in range(750):
        error = nn.train(X,y,50)
        print "step %i training error: %f" % (i+1, error)
    with open("conv_net.dat","wb") as f:
        pickle.dump(nn,f,pickle.HIGHEST_PROTOCOL)
        
print "finding dory in scenes"
scene_path = glob.glob('Scenes/*.PNG')
scenes = [Image.open(img) for img in scene_path]

scan_sizes = [500,450,400,350,300,250,200]

for s in range(len(scenes)):
    print "scanning scene %i of %i" % (s+1, len(scenes))
    max_prob = 0
    for scan in scan_sizes:
        for i in range(0,scenes[s].size[0]-scan,25):
            for j in range(0,scenes[s].size[1]-scan,25):
                img = scenes[s].crop((i,j,i+scan,j+scan))
                img = img.resize((200,200), Image.ANTIALIAS)
                img = np.array(img,dtype='float64')/256
                img = img.transpose(2, 0, 1).reshape(1,4,200,200)
                probs,pred = nn.predict(img)
                if probs[0,1] > max_prob:
                    dims = (i,j,i+scan,j+scan)
                    max_prob = probs[0,1]
    box = ImageDraw.Draw(scenes[s])
    box.rectangle(dims,outline=255)
    box.rectangle((dims[0]-1,dims[1]-1,dims[2]+1,dims[3]+1),outline=255)
    box.rectangle((dims[0]-2,dims[1]-2,dims[2]+2,dims[3]+2),outline=255)
    scenes[s].show()
    scenes[s].save('scene%i.jpg' % (int(s)+1),quality=100)