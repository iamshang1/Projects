import numpy as np
import theano
import theano.tensor as T
import gzip, cPickle
import sys
import matplotlib.pyplot as plt

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train = np.array(train_set[0])
y_train = np.array(train_set[1])
X_test = np.array(test_set[0])
y_test = np.array(test_set[1])

class neural_network(object):
    def __init__(self,hidden_layers,layer_nodes):
        self.input = T.matrix()
        self.target = T.matrix()
        self.W = []
        self.b = []
        self.lin_outputs = []
        self.batch_norms = []
        self.gammas = []
        self.betas = []
        self.activations = []
        self.W.append(theano.shared(self.ortho_weight(784,layer_nodes),borrow=True))
        self.b.append(theano.shared(np.zeros((layer_nodes,), dtype=theano.config.floatX),borrow=True))
        self.gammas.append(theano.shared(value = np.ones((layer_nodes,), dtype=theano.config.floatX)))
        self.betas.append(theano.shared(value = np.zeros((layer_nodes,), dtype=theano.config.floatX)))
        self.lin_outputs.append(T.dot(self.input,self.W[-1])+self.b[-1])
        self.batch_norms.append(T.nnet.bn.batch_normalization(self.lin_outputs[-1],gamma=self.gammas[-1],beta=self.betas[-1],
            mean=T.mean(self.lin_outputs[-1], axis=0),std=T.sqrt(T.var(self.lin_outputs[-1], axis=0)+0.00001)))
        self.activations.append(T.nnet.sigmoid(self.batch_norms[-1]))
        for layer in range(hidden_layers-1):
            self.W.append(theano.shared(self.ortho_weight(layer_nodes,layer_nodes),borrow=True))
            self.b.append(theano.shared(np.zeros((layer_nodes,), dtype=theano.config.floatX),borrow=True))
            self.gammas.append(theano.shared(value = np.ones((layer_nodes,), dtype=theano.config.floatX)))
            self.betas.append(theano.shared(value = np.zeros((layer_nodes,), dtype=theano.config.floatX)))
            self.lin_outputs.append(T.dot(self.activations[-1],self.W[-1])+self.b[-1])
            self.batch_norms.append(T.nnet.bn.batch_normalization(self.lin_outputs[-1],gamma=self.gammas[-1],beta=self.betas[-1],
                mean=T.mean(self.lin_outputs[-1], axis=0),std=T.sqrt(T.var(self.lin_outputs[-1], axis=0)+0.00001)))
            self.activations.append(T.nnet.sigmoid(self.batch_norms[-1]))
        self.W.append(theano.shared(self.ortho_weight(layer_nodes,10),borrow=True))
        self.b.append(theano.shared(np.zeros((10,), dtype=theano.config.floatX),borrow=True))
        self.gammas.append(theano.shared(value = np.ones((10,), dtype=theano.config.floatX)))
        self.betas.append(theano.shared(value = np.zeros((10,), dtype=theano.config.floatX)))
        self.lin_outputs.append(T.dot(self.activations[-1],self.W[-1])+self.b[-1])
        self.batch_norms.append(T.nnet.bn.batch_normalization(self.lin_outputs[-1],gamma=self.gammas[-1],beta=self.betas[-1],
            mean=T.mean(self.lin_outputs[-1], axis=0),std=T.sqrt(T.var(self.lin_outputs[-1], axis=0)+0.00001)))
        self.activations.append(T.nnet.sigmoid(self.batch_norms[-1]))
        self.cost = T.nnet.categorical_crossentropy(self.activations[-1],self.target).mean()
        self.params = self.W+self.b+self.gammas+self.betas
        self.updates = self.adam(self.cost,self.params)
        self.train_f = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.predict_f = theano.function([self.input],self.activations[-1],allow_input_downcast=True)

    def ortho_weight(self,fan_in,fan_out):
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W)
        if fan_in > fan_out:
            W = u[:fan_in,:fan_out]
        else:
            W = v[:fan_in,:fan_out]
        return W.astype(theano.config.floatX)
        
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
            X = X[indices,:]
            y = y[indices]
        target = np.zeros((y.shape[0],len(np.unique(y))))
        for i in range(len(np.unique(y))):
            target[y==i,i] = 1
        return self.train_f(X,target)

    def predict(self,X,y=None):
        prediction = self.predict_f(X)
        return np.argmax(prediction,axis=1)
        
batch_size = 2000
nn = neural_network(4,1000)

test_error = []

for i in range(3000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i training error: %f \r" % (i+1, cost))
    sys.stdout.flush()
    pred = nn.predict(X_test)
    test_error.append(1-float(np.sum(pred==y_test))/len(pred))

cPickle.dump(test_error, open('batch_norm_accuracy.p', 'wb'))

plt.scatter(range(len(test_error)),test_error,alpha=0.5)
plt.title("Test Set Accuracy")
plt.xlabel('Iteration')
plt.ylabel('Test Set Error')
plt.show()