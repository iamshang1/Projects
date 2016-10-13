import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

data = make_classification(3000,2,2,0,random_state=2)
X = data[0]
y = data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

features = 2
learning_rate = 0.1
regularization = 0.001
hidden_layers = 1
nodes_per_hidden_layer = 50
classes = 2
batch_size = 200

class layer(object):
    def __init__(self,features,nodes,):
        self.bound = np.sqrt(1./(features+nodes))
        self.w = theano.shared(np.asarray(np.random.uniform(low=-self.bound,high=self.bound,size=(features,nodes)),dtype=theano.config.floatX))
        self.b = theano.shared(np.asarray(np.random.uniform(low=-.5, high=.5, size=(nodes)),dtype=theano.config.floatX))
    def get_params(self):
        return self.w,self.b

class neural_network(object):
    def __init__(self,classes,hidden_layers,features,nodes_per_hidden_layer,learning_rate,regularization):
        self.hidden_layers = []
        self.hidden_layers.append(layer(features,nodes_per_hidden_layer))
        for i in range(hidden_layers-1):
            self.hidden_layers.append(layer(nodes_per_hidden_layer,nodes_per_hidden_layer))
        self.output_layer = layer(nodes_per_hidden_layer,classes)
        self.params = []
        for l in self.hidden_layers:
            self.params.extend(l.get_params())
        self.params.extend(self.output_layer.get_params())
        self.A = T.matrix()
        self.t = T.matrix()
        self.s = 1/(1+T.exp(-T.dot(self.A,self.params[0])-self.params[1]))
        for i in range(hidden_layers):
            self.s = 1/(1+T.exp(-T.dot(self.s,self.params[2*(i+1)])-self.params[2*(i+1)+1]))
        self.cost = -self.t*T.log(self.s)-(1-self.t)*T.log(1-self.s)
        self.cost = self.cost.mean()
        for i in range(hidden_layers+1):
            self.cost += regularization*(self.params[2*i]**2).mean()
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        self.propogate = theano.function([self.A,self.t],self.cost,updates=[(param,param-learning_rate*gparam) for param,gparam in zip(self.params,self.gparams)],allow_input_downcast=True)
        self.classify = theano.function([self.A],self.s,allow_input_downcast=True)
    def train(self,X,y,batch_size=None):
        if batch_size:
            indices = np.random.permutation(X.shape[0])[:batch_size]
            X = X[indices,:]
            y = y[indices]
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        return self.propogate(X,target)
    def predict(self,X):
        prediction = self.classify(X)
        return np.argmax(prediction,axis=1)

test = neural_network(classes,hidden_layers,features,nodes_per_hidden_layer,learning_rate,regularization)
for i in range(20000):
    print "step %i training error:" % (i+1)
    print test.train(X_train,y_train,batch_size)
    if (i+1)%500 == 0:
        h = 0.2
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
        Z = test.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
        plt.axis('off')
        color_map = {0: (0, 0, .9), 1: (.8, .6, 0)}
        colors = [color_map[y] for y in y_train]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, cmap=plt.cm.Paired)
        plt.title("Decision Boundary After %i Iterations" % (i+1))
        plt.savefig("%i.png" % (i+1))