import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
try:
    import cPickle as pickle
except:
    import pickle
import sys
from sklearn.model_selection import train_test_split

sys.setrecursionlimit(10000)

class conv_lstm(object):
    '''
    convolutional long short term memory network for classifying human activity 
    from raw accelerometer data
    
    parameters:
      - binary: boolean (default False)
        use True if labels are for ambulatory/non-ambulatory
        use False if labels are for non-ambulatory/walking/running/upstairs/downstairs
        
    methods:
      - train(X_train, y_train)
        train conv-lstm network
        parameters:
          - X_train: 4d numpy array
            training features output by record_fetcher_between_subject_raw.py
          - y_train: 2d numpy array
            training labels output by record_fetcher_between_subject_raw.py
        outputs:
          - training loss at given iteration
      - predict(X_test)
        predict label from test data
        parameters:
          - X_test: 4d numpy array
            testing features output by record_fetcher_between_subject_raw.py
        outputs:
          - predicted labels for test features
    '''
    def __init__(self,binary=False):
        self.input = T.tensor4()
        
        #layer 1 - convolution
        self.conv_w1 = theano.shared(self.ortho_weight_conv(64,1,10,10),borrow=True)
        self.conv_b1 = theano.shared(np.zeros((64,), dtype=theano.config.floatX),borrow=True)
        self.conv_out1 = conv2d(input=self.input, filters=self.conv_w1, subsample=(5,1), border_mode='valid')
        self.layer_out1 = T.nnet.elu(self.conv_out1 + self.conv_b1.dimshuffle('x', 0, 'x', 'x'))
        
        #layer 2 - convolution
        self.conv_w2 = theano.shared(self.ortho_weight_conv(64,64,3,1),borrow=True)
        self.conv_b2 = theano.shared(np.zeros((64,), dtype=theano.config.floatX),borrow=True)
        self.conv_out2 = conv2d(input=self.layer_out1, filters=self.conv_w2, subsample=(2,1), border_mode='valid')
        self.layer_out2 = T.nnet.elu(self.conv_out2 + self.conv_b2.dimshuffle('x', 0, 'x', 'x'))
        
        #layer 3 - convolution
        self.conv_w3 = theano.shared(self.ortho_weight_conv(64,64,3,1),borrow=True)
        self.conv_b3 = theano.shared(np.zeros((64,), dtype=theano.config.floatX),borrow=True)
        self.conv_out3 = conv2d(input=self.layer_out2, filters=self.conv_w3, subsample=(2,1), border_mode='valid')
        self.layer_out3 = T.nnet.elu(self.conv_out3 + self.conv_b3.dimshuffle('x', 0, 'x', 'x'))
        
        #layer 4 - convolution
        self.conv_w4 = theano.shared(self.ortho_weight_conv(64,64,3,1),borrow=True)
        self.conv_b4 = theano.shared(np.zeros((64,), dtype=theano.config.floatX),borrow=True)
        self.conv_out4 = conv2d(input=self.layer_out3, filters=self.conv_w4, subsample=(2,1), border_mode='valid')
        self.layer_out4 = T.nnet.elu(self.conv_out4 + self.conv_b4.dimshuffle('x', 0, 'x', 'x'))
        
        #flatten convolutional output for lstm layer
        self.flattened = self.layer_out4.reshape((64,24)).T
        
        #layer 5 - lstm units
        self.Wi = theano.shared(self.ortho_weight(64)[:,:32],borrow=True)
        self.Wf = theano.shared(self.ortho_weight(64)[:,:32],borrow=True)
        self.Wc = theano.shared(self.ortho_weight(64)[:,:32],borrow=True)
        self.Wo = theano.shared(self.ortho_weight(64)[:,:32],borrow=True)
        self.Ui = theano.shared(self.ortho_weight(32),borrow=True)
        self.Uf = theano.shared(self.ortho_weight(32),borrow=True)
        self.Uc = theano.shared(self.ortho_weight(32),borrow=True)
        self.Uo = theano.shared(self.ortho_weight(32),borrow=True)
        self.bi = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        self.bf = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        self.bc = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        self.bo = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        self.C0 = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        self.h0 = theano.shared(np.zeros((32,), dtype=theano.config.floatX),borrow=True)
        
        #layer 6 - softmax
        if binary:
            self.W2 = theano.shared(self.ortho_weight(32)[:,:2],borrow=True)
            self.b2 = theano.shared(np.zeros((2,), dtype=theano.config.floatX),borrow=True)
        else:
            self.W2 = theano.shared(self.ortho_weight(32)[:,:5],borrow=True)
            self.b2 = theano.shared(np.zeros((5,), dtype=theano.config.floatX),borrow=True)
        self.target = T.matrix()
        
        #scan operation for lstm layer
        self.params1 = [self.Wi,self.Wf,self.Wc,self.Wo,self.Ui,self.Uf,self.Uc,self.Uo,self.bi,self.bf,self.bc,self.bo]
        [self.c1,self.h_output1],_ = theano.scan(fn=self.step,sequences=self.flattened,outputs_info=[self.C0,self.h0],non_sequences=self.params1)
        
        #final softmax, final output is average of output of last 4 timesteps
        self.output = T.nnet.softmax(T.dot(self.h_output1,self.W2)+self.b2)[-4:,:]
        self.output = T.mean(self.output,0,keepdims=True)
        
        #cost, updates, train, and predict
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.fullparams = self.params1 + [self.conv_w1,self.conv_b1,self.conv_w2,self.conv_b2,self.conv_w3,self.conv_b3,self.conv_w4,self.conv_b4,self.h0,self.C0,self.W2,self.b2]
        self.updates = self.adam(self.cost,self.fullparams)
        self.train = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.predict = theano.function([self.input],self.output,allow_input_downcast=True)

    def step(self,input,h0,C0,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo):
        '''
        step function for lstm unit
        '''
        i = T.nnet.sigmoid(T.dot(input,Wi)+T.dot(h0,Ui)+bi)
        cand = T.tanh(T.dot(input,Wc)+T.dot(h0,Uc)+bc)
        f = T.nnet.sigmoid(T.dot(input,Wf)+T.dot(h0,Uf)+bf)
        c = cand*i+C0*f
        o = T.nnet.sigmoid(T.dot(input,Wo)+T.dot(h0,Uo)+bo)
        h = o*T.tanh(c)
        return c,h

    def ortho_weight(self,ndim):
        '''
        orthogonal weight initialization for lstm layers
        '''
        bound = np.sqrt(1./ndim)
        W = np.random.randn(ndim, ndim)*bound
        u, s, v = np.linalg.svd(W)
        return u.astype(theano.config.floatX)
        
    def ortho_weight_conv(self,chan_out,chan_in,filter_h,filter_w):
        '''
        orthogonal weight initialization for convolutional layers
        '''
        bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
        W = np.random.random((chan_out, chan_in * filter_h * filter_w))
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u.reshape((chan_out, chan_in, filter_h, filter_w))
        else:
            W = v.reshape((chan_out, chan_in, filter_h, filter_w))
        return W.astype(theano.config.floatX)

    def adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.01, e=1e-8):
        '''
        adaptive moment estimation gradient descent
        '''
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

#load data
X_train = np.load('X_train_raw.npy')
X_test = np.load('X_test_raw.npy')
y_train = np.load('y_train_raw.npy')
y_test = np.load('y_test_raw.npy')

# verify the required arguments are given
if (len(sys.argv) < 2):
    print 'Usage: python convlstm_between_subject.py <1 for 2-category labels, 0 for 5-category labels>'
    exit(1)

if sys.argv[1] == '1':
    binary = True
elif sys.argv[1] == '0':
    binary = False
else:
    print 'Usage: python convlstm_between_subject.py <1 for 2-category labels, 0 for 5-category labels>'
    exit(1)

#train    
NN = conv_lstm(binary=binary)

for i in range(1000000):
    #select random training sample
    idx = np.random.randint(y_train.shape[0])
    X_in = np.expand_dims(X_train[idx,:,:], 0)
    y_in = np.expand_dims(y_train[idx,:],0)
    
    #train on random sample
    cost = NN.train(X_in,y_in)
    print "step %i training error: %.4f \r" % (i+1, cost),
    
    #predict every 10000 iterations
    if (i+1) % 10000 == 0:
        correct = 0
        
        #predict each entry in test set
        for j in range(y_test.shape[0]):
            print "predicting %i of %i in test set \r" % (j+1, y_test.shape[0]),
            pred = NN.predict(np.expand_dims(X_test[j,:,:],0))
            if np.argmax(pred[0]) == np.argmax(y_test[j,:]):
                correct += 1
        print "step %i test accuracy: %.4f           " % (i+1,float(correct)/y_test.shape[0]*100)