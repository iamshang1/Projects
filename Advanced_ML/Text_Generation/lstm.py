import numpy as np
import theano
import theano.tensor as T
try:
    import cPickle as pickle
except:
    import pickle
import sys

dic = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',\
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',\
    '1','2','3','4','5','6','7','8','9','0','-','.',',','!','?','(',')','\'','"',' ']

text = np.load('hhgttg.npy')
batch = 160

sys.setrecursionlimit(10000)

class lstm(object):
    def __init__(self):
        self.input = T.matrix()
        self.Wi = theano.shared(self.ortho_weight(256)[:72,:])
        self.Wf = theano.shared(self.ortho_weight(256)[:72,:])
        self.Wc = theano.shared(self.ortho_weight(256)[:72,:])
        self.Wo = theano.shared(self.ortho_weight(256)[:72,:])
        self.Ui = theano.shared(self.ortho_weight(256))
        self.Uf = theano.shared(self.ortho_weight(256))
        self.Uc = theano.shared(self.ortho_weight(256))
        self.Uo = theano.shared(self.ortho_weight(256))
        self.bi = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.bf = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.bc = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.bo = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.C0 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.h0 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(256)),dtype=theano.config.floatX))
        self.W2 = theano.shared(self.ortho_weight(256)[:,:72])
        self.b2 = theano.shared(np.asarray(np.random.uniform(low=-0.1,high=0.1,size=(72)),dtype=theano.config.floatX))
        self.target = T.matrix()
        
        self.params = [self.h0,self.C0,self.Wi,self.Wf,self.Wc,self.Wo,self.Ui,self.Uf,self.Uc,self.Uo,self.bi,self.bf,self.bc,self.bo]
        [self.c,self.h_output],_ = theano.scan(fn=self.step,sequences=self.input,outputs_info=[self.C0,self.h0],non_sequences=self.params[:-4])
        self.output =  T.nnet.softmax(T.dot(self.h_output,self.W2)+self.b2)[40:,:]
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.updates = self.adam(self.cost,self.params+[self.W2,self.b2])
        self.train = theano.function([self.input,self.target],self.cost,updates=self.updates,allow_input_downcast=True)
        self.predict = theano.function([self.input],self.output,allow_input_downcast=True)

    def step(self,input,h0,C0,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo):
        i = T.nnet.sigmoid(T.dot(input,Wi)+T.dot(h0,Ui)+bi)
        cand = T.tanh(T.dot(input,Wc)+T.dot(h0,Uc)+bc)
        f = T.nnet.sigmoid(T.dot(input,Wf)+T.dot(h0,Uf)+bf)
        c = cand*i+C0*f
        o = T.nnet.sigmoid(T.dot(input,Wo)+T.dot(h0,Uo)+bo)
        h = o*T.tanh(c)
        return c,h

    def ortho_weight(self,ndim):
        bound = np.sqrt(1./ndim)
        W = np.random.randn(ndim, ndim)*bound
        u, s, v = np.linalg.svd(W)
        return u.astype(theano.config.floatX)

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

try:
    with open("nn_cost.dat","rb") as c:
        min_cost = float(c.readline())
except:
    min_cost = 1000

try:
    with open("nn.dat","rb") as pickle_nn:
        NN = pickle.load(pickle_nn)
    print "saved neural network loaded"        
except:
    print "pickle load failed, creating new NN"
    NN = lstm()
    with open("nn.dat","wb") as f:
        pickle.dump(NN,f,pickle.HIGHEST_PROTOCOL)

#train
generated = open('generated.txt','w')
for i in range(1000000):
    start = np.random.randint(40,text.shape[0]-batch-1)
    X_train = text[start-40:start+batch,:]
    y_train = text[start+1:start+1+batch,:]
    cost = NN.train(X_train,y_train)
    print "step %i training error:" % (i+1), cost
    if (i+1) % 500 == 0:
        string = ""
        start = np.random.randint(text.shape[0]-41)
        X_test = text[start:start+41,:]
        for j in range(40):
            row = X_test[j,:]
            max = np.argmax(row)
            string += dic[max]
        out = NN.predict(X_test)
        for j in range(160):
            max = np.argmax(out)
            string += dic[max]
            next = np.zeros((1,72))
            next[0,max] = 1
            X_test = np.vstack((X_test[1:,:],next))
            out = NN.predict(X_test)
        print string
        generated.write((str(i+1)+": "+string+"\n"))
        if cost < min_cost:
            print "pickling neural network"
            min_cost = cost
            with open("nn_cost.dat","wb") as c:
                c.write(str(min_cost))
            with open("nn.dat","wb") as f:
                pickle.dump(NN,f,pickle.HIGHEST_PROTOCOL)
        elif cost < min_cost+0.2:
            print "pickling neural network"
            with open("nn_cost.dat","wb") as c:
                c.write(str(min_cost))
            with open("nn.dat","wb") as f:
                pickle.dump(NN,f,pickle.HIGHEST_PROTOCOL)
        elif cost > min_cost+0.5:
            print "reloading last good NN"
            with open("nn.dat","rb") as pickle_nn:
                NN = pickle.load(pickle_nn)
generated.close()