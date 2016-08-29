import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import zipfile
import gzip, cPickle
import timeit
import sys
try:
    import cPickle as pickle
except:
    import pickle
import matplotlib.pyplot as plt

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train = np.array(train_set[0])
y_train = np.array(train_set[1])
X_test = np.array(test_set[0])
y_test = np.array(test_set[1])

class RBM(object):
    def __init__(self,n_visible,n_hidden,batch_size):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        np_rng = np.random.RandomState(1234)
        theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        initial_W = np.asarray(np_rng.uniform(low=-4*np.sqrt(6./(n_hidden + n_visible)),high=4*np.sqrt(6./(n_hidden+n_visible)),
            size=(n_visible, n_hidden)),dtype=theano.config.floatX)
        W = theano.shared(value=initial_W, name='W', borrow=True)

        hbias = theano.shared(value=np.zeros(n_hidden,dtype=theano.config.floatX),name='hbias',borrow=True)
        vbias = theano.shared(value=np.zeros(n_visible,dtype=theano.config.floatX),name='vbias',borrow=True)

        self.input = T.matrix('input')
        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        self.params = [self.W, self.hbias, self.vbias]
        
        self.persistent_chain = theano.shared(np.zeros((batch_size,n_hidden),dtype=theano.config.floatX),borrow=True)
        self.cost, self.updates = self.get_cost_updates(lr=0.1,persistent=self.persistent_chain, k=15)
        self.train = theano.function([self.input],self.cost,updates=self.updates,name='train_rbm')
        
        _i,_j,self._output_hidden = self.sample_h_given_v(self.input)
        self.output_hidden = theano.function([self.input],self._output_hidden)
        
    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1 = T.dot(v0_sample, self.W) + self.hbias
        h1_mean = T.nnet.sigmoid(pre_sigmoid_h1)
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,n=1,p=h1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]
        
    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1 = T.dot(h0_sample, self.W.T) + self.vbias
        v1_mean = T.nnet.sigmoid(pre_sigmoid_v1)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,n=1,p=v1_mean,dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]
        
    def gibbs_step(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]
        
    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term
        
    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        ([pre_sigmoid_nvs,nv_means,nv_samples,pre_sigmoid_nhs,nh_means,nh_samples],updates) = \
            theano.scan(self.gibbs_step, outputs_info=[None, None, None, None, None, chain_start],n_steps=k,name="gibbs_step")
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr,dtype=theano.config.floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)  
        else:
            monitoring_cost = self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])
        return monitoring_cost, updates
        
    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

class NN(object):
    def __init__(self,rbm1,rbm2,rbm3,rbm4):
        self.learning_rate = 0.01
        self.W1 = rbm1.W
        self.W2 = rbm2.W
        self.W3 = rbm3.W
        self.W4 = rbm4.W        
        self.W5 = theano.shared(self.ortho_weight(1000,10),borrow=True)
        self.b1 = rbm1.hbias
        self.b2 = rbm2.hbias
        self.b3 = rbm3.hbias
        self.b4 = rbm4.hbias
        self.b5 = (theano.shared(np.zeros((10,), dtype=theano.config.floatX),borrow=True))
        self.input = T.matrix()
        self.target = T.matrix()
        
        self.l1out = T.nnet.sigmoid(T.dot(self.input,self.W1)+self.b1)
        self.l2out = T.nnet.sigmoid(T.dot(self.l1out,self.W2)+self.b2)
        self.l3out = T.nnet.sigmoid(T.dot(self.l2out,self.W3)+self.b3)
        self.l4out = T.nnet.sigmoid(T.dot(self.l3out,self.W4)+self.b4)
        self.output = T.nnet.softmax(T.dot(self.l4out,self.W5)+self.b5)
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.params = [self.W1,self.W2,self.W3,self.W4,self.W5,self.b1,self.b2,self.b3,self.b4,self.b5]
        self.updates = self.adam(self.cost,self.params)
        self.train_f = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.predict_f = theano.function([self.input],self.output,allow_input_downcast=True)

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

    def predict(self,X):
        prediction = self.predict_f(X)
        return np.argmax(prediction,axis=1)
      
batch_size = 20
n_train_batches = X_train.shape[0]//batch_size

try:
    with open("rbm1.dat","rb") as pickle_nn:
        rbm1 = pickle.load(pickle_nn)
    print "saved rbm1 loaded"        
except:
    print "pickle load failed, creating new rbm1"
    rbm1 = RBM(784,1000,batch_size)
    start_time = timeit.default_timer()
    for i in range(15):
        mean_cost = []
        for idx in range(n_train_batches):
            X_in = X_train[idx:idx+batch_size,:]
            mean_cost += [rbm1.train(X_in)]
        print 'RBM1 iteration %i cost:' % (i+1), np.mean(mean_cost)
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('RBM1 training took %f minutes' % (pretraining_time/60.))
    with open("rbm1.dat","wb") as f:
        pickle.dump(rbm1,f,pickle.HIGHEST_PROTOCOL)

try:
    with open("rbm2.dat","rb") as pickle_nn:
        rbm2 = pickle.load(pickle_nn)
    print "saved rbm2 loaded"
except:
    print "pickle load failed, creating new rbm2"
    rbm2 = RBM(1000,1000,batch_size)
    start_time = timeit.default_timer()
    for i in range(15):
        mean_cost = []
        for idx in range(n_train_batches):
            X_in = rbm1.output_hidden(X_train[idx:idx+batch_size,:])
            mean_cost += [rbm2.train(X_in)]
        print 'RBM2 iteration %i cost:' % (i+1), np.mean(mean_cost)
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('RBM2 training took %f minutes' % (pretraining_time/60.))
    with open("rbm2.dat","wb") as f:
        pickle.dump(rbm2,f,pickle.HIGHEST_PROTOCOL)

try:
    with open("rbm3.dat","rb") as pickle_nn:
        rbm3 = pickle.load(pickle_nn)
    print "saved rbm3 loaded"
except:
    print "pickle load failed, creating new rbm3"
    rbm3 = RBM(1000,1000,batch_size)
    start_time = timeit.default_timer()
    for i in range(15):
        mean_cost = []
        for idx in range(n_train_batches):
            X_in1 = rbm1.output_hidden(X_train[idx:idx+batch_size,:])
            X_in2 = rbm2.output_hidden(X_in1)
            mean_cost += [rbm3.train(X_in2)]
        print 'RBM3 iteration %i cost:' % (i+1), np.mean(mean_cost)
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('RBM3 training took %f minutes' % (pretraining_time/60.))
    with open("rbm3.dat","wb") as f:
        pickle.dump(rbm3,f,pickle.HIGHEST_PROTOCOL)

try:
    with open("rbm4.dat","rb") as pickle_nn:
        rbm4 = pickle.load(pickle_nn)
    print "saved rbm4 loaded"
except:
    print "pickle load failed, creating new rbm4"
    rbm4 = RBM(1000,1000,batch_size)
    start_time = timeit.default_timer()
    for i in range(15):
        mean_cost = []
        for idx in range(n_train_batches):
            X_in1 = rbm1.output_hidden(X_train[idx:idx+batch_size,:])
            X_in2 = rbm2.output_hidden(X_in1)
            X_in3 = rbm3.output_hidden(X_in2)
            mean_cost += [rbm4.train(X_in3)]
        print 'RBM4 iteration %i cost:' % (i+1), np.mean(mean_cost)
    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time)
    print ('RBM4 training took %f minutes' % (pretraining_time/60.))
    with open("rbm4.dat","wb") as f:
        pickle.dump(rbm4,f,pickle.HIGHEST_PROTOCOL)
        
batch_size = 2000
nn = NN(rbm1,rbm2,rbm3,rbm4)

test_error = []

for i in range(3000):
    cost = nn.train(X_train,y_train,batch_size)
    sys.stdout.write("step %i training error: %f \r" % (i+1, cost))
    sys.stdout.flush()
    pred = nn.predict(X_test)
    test_error.append(1-float(np.sum(pred==y_test))/len(pred))

cPickle.dump(test_error, open('rbm_pretraining_accuracy.p', 'wb'))

plt.scatter(range(len(test_error)),test_error,alpha=0.5)
plt.title("Test Set Accuracy")
plt.xlabel('Iteration')
plt.ylabel('Test Set Error')
plt.show()