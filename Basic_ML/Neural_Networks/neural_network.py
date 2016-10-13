import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

data = make_classification(3000,2,2,0)
X = data[0]
y = data[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

class neuron(object):
    def __init__(self,activation):
        self.activation = np.concatenate((np.ones((activation.shape[0],1)),activation),axis=1)
        self.weights = np.random.rand(self.activation.shape[1])
    def logistic_regression(self):
        a = np.dot(self.activation,self.weights.T)
        return 1/(1+np.exp(-a))
    def update_activation(self,activation):
        self.activation = np.concatenate((np.ones((activation.shape[0],1)),activation),axis=1)
    def update_weights(self,node_error,learning_rate,regularization):
        for i in range(self.weights.shape[0]):
            self.weights[i] -= learning_rate*np.mean(node_error*self.activation[:,i])
            self.weights[i] -= learning_rate*regularization*self.weights[i]
    def get_weights(self):
        return self.weights

class layer(object):
    def __init__(self,nodes,activation,regularization):
        self.activation = activation
        self.regularization = regularization
        self.nodes = [neuron(self.activation) for i in range(nodes)]
    def forward_propogate(self,activation):
        self.outputs = np.empty((activation.shape[0],len(self.nodes)))
        for i in range(len(self.nodes)):
            self.outputs[:,i] = self.nodes[i].logistic_regression()
        return self.outputs
    def back_propogate(self,errors,learning_rate,weights=None):
        new_errors = []
        for i in range(len(self.nodes)):
            if weights!=None:
                node_error = np.dot(errors,weights[:,i+1])*self.outputs[:,i]*(1-self.outputs[:,i])
            else:
                node_error = errors[:,i]*self.outputs[:,i]*(1-self.outputs[:,i])
            new_errors.append(node_error)
            self.nodes[i].update_weights(node_error,learning_rate,self.regularization)
        return np.array((new_errors)).T
    def get_weights(self):
        weights = []
        for node in self.nodes:
            weights.append(node.get_weights())
        weights = np.array((weights))
        return weights
        
class neural_network(object):
    def __init__(self,input_features,input_classes,hidden_layers,nodes_per_layer,classes,learning_rate,regularization):
        self.input_features = input_features
        self.input_classes = input_classes
        self.num_hidden_layers = hidden_layers
        self.nodes_per_layer = nodes_per_layer
        self.classes = classes
        self.learning_rate = learning_rate
        self.regularization = regularization 
        self.network = self.create_network()
    def create_network(self):
        self.hidden_layers = [layer(self.nodes_per_layer,self.input_features,self.regularization)]
        for i in range(self.num_hidden_layers-1):
            self.hidden_layers.append(layer(self.nodes_per_layer,np.ones((100,self.nodes_per_layer)),self.regularization))
        self.output_layer = layer(self.classes,np.ones((100,self.nodes_per_layer)),self.regularization)
    def forward_propogation(self):
        activation = np.copy(self.input_features)
        for layer in self.hidden_layers:
            for node in layer.nodes:
                node.update_activation(activation)
            activation = layer.forward_propogate(activation)
        for node in self.output_layer.nodes:
            node.update_activation(activation)
        self.class_probabilities = self.output_layer.forward_propogate(activation)
    def back_propogation(self):
        errors = np.zeros((self.input_classes.shape[0],self.classes))
        for i in range(self.classes):
            errors[self.input_classes==i,i] = 1
        self.errors = errors = self.class_probabilities-errors
        weights = []
        weights.append(self.output_layer.get_weights())
        errors = self.output_layer.back_propogate(errors,self.learning_rate)
        for layer in reversed(self.hidden_layers):
            weights.append(layer.get_weights())
            errors = layer.back_propogate(errors,self.learning_rate,weights[-2])
    def classification_accuracy(self):
        self.prediction = np.argmax(self.class_probabilities,axis=1)
        correct = self.prediction == self.input_classes
        print "classification accuracy: %f" % (float(np.sum(correct))/np.shape(correct)[0])
        print "mean error: %f" % np.mean(self.errors)
    def train(self,iterations):
        for i in range(iterations):
            print "iteration %i" % (i+1)
            self.forward_propogation()
            self.back_propogation()
            self.classification_accuracy()
    def predict(self,X):
        activation = X
        for layer in self.hidden_layers:
            for node in layer.nodes:
                node.update_activation(activation)
            activation = layer.forward_propogate(activation)
        for node in self.output_layer.nodes:
            node.update_activation(activation)
        self.predict_probabilities = self.output_layer.forward_propogate(activation)
        return np.argmax(self.predict_probabilities,axis=1)
    def predict_accuracy(self,y):
        self.prediction = np.argmax(self.predict_probabilities,axis=1)
        correct = self.prediction == y
        print "test set classification accuracy: %f" % (float(np.sum(correct))/np.shape(correct)[0])
        print "test set mean error: %f" % np.mean(self.errors)
        
test = neural_network(X_train,y_train,1,40,2,1,0.0)
for i in range(20000):
    print "iteration %i" % (i+1)
    test.forward_propogation()
    test.back_propogation()
    test.classification_accuracy()
    if (i+1)%1000 == 0:
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
        plt.show()
test.predict(X_test)
test.predict_accuracy(y_test)