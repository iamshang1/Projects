import glob
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from PIL import Image
from sklearn.cross_validation import train_test_split

print "loading images"
image_path = glob.glob('SimpleImageDataset/*.jpg')
images = [Image.open(path) for path in image_path]
images = [img.resize((96,128), Image.ANTIALIAS) for img in images]
input = np.empty((0,3,96,128))
for image in images:
    img = np.array(image,dtype='float64')/256
    img = img.transpose(2, 0, 1).reshape(1,3,96,128)
    input = np.concatenate((input,img),axis=0)
labels = []
for i in range(len(image_path)):
    if image_path[i][:-len('00.jpg')] == "SimpleImageDataset\\building":
        labels.append(0)
    elif image_path[i][:-len('00.jpg')] == "SimpleImageDataset\scene":
        labels.append(1)
    elif image_path[i][:-len('00.jpg')] == "SimpleImageDataset\\text":
        labels.append(2)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

learning_rate = 0.001
regularization = 0.001
convolutional_layers = 2
feature_maps = [3,5,5]
filter_shapes = [(10,10),(10,10)]
poolsize = [(2,2),(2,2)]
feedforward_layers = 1
feedforward_nodes = [100]
classes = 3

flattened = list(input.shape[2:])
for i in range(convolutional_layers):
    flattened[0] = flattened[0] - filter_shapes[i][0] + 1
    flattened[1] = flattened[1] - filter_shapes[i][1] + 1
    flattened[0] = flattened[0]/poolsize[i][0]
    flattened[1] = flattened[1]/poolsize[i][1]
flattened = np.prod(flattened)
flattened *= feature_maps[-1]

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
    def __init__(self,convolutional_layers,feature_maps,filter_shapes,poolsize,feedforward_layers,feedforward_nodes,classes,learning_rate,regularization):
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
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        self.propogate = theano.function([self.input,self.target],self.cost,updates=[(param,param-learning_rate*gparam) for param,gparam in zip(self.params,self.gparams)],allow_input_downcast=True)
        self.classify = theano.function([self.input],self.output,allow_input_downcast=True)
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
        return np.argmax(prediction,axis=1)

print "building neural network"
nn = neural_network(convolutional_layers,feature_maps,filter_shapes,poolsize,feedforward_layers,feedforward_nodes,classes,learning_rate,regularization)

for i in range(5000):
    error = nn.train(X_train,y_train)
    print "step %i training error: %f" % (i+1, error)
    if (i+1)%100 == 0:
        pred = nn.predict(X_test)
        accuracy = float(np.sum(pred==y_test))/len(y_test)
        print "test set prediction accuracy after %i iterations: %f" % (i+1,accuracy)