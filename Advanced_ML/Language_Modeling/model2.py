from sklearn.preprocessing import StandardScaler
import numpy as np
import theano
import theano.tensor as T
import gensim
import string
import re
import sys

sys.setrecursionlimit(10000)

#load hgttg corpus
print "loading dataset"
with open('hhgttg.txt','r') as f:
    dataset = f.read()

#convert dataset to list of sentences
print "converting dataset to list of sentences"
dataset = re.sub(r'-|\t|\n',' ',dataset)
sentences = dataset.split('.')
sentences = [sentence.translate(None, string.punctuation).lower().split() for sentence in sentences]

#save copy of cleaned tokenized dataset for training lstm
dataset = dataset.replace (".", " ")
dataset = dataset.translate(None, string.punctuation).lower().split()

#train word2vec
print "training word2vec"
w2v = gensim.models.Word2Vec(sentences, size=100, min_count=1, workers=4, iter=10, sample=0)
w2v.init_sims(replace=True)

#save all word embeddings to matrix
print "saving word vectors to matrix"
vocab = np.empty((len(w2v.vocab),100))
word2id = {}
for key,val in w2v.vocab.iteritems():
    idx = val.__dict__['index']
    vocab[idx,:] = w2v[key]
    word2id[key] = idx
id2word = dict(zip(word2id.values(), word2id.keys()))

#normalize word vectors
scaler = StandardScaler()
vocab = scaler.fit_transform(vocab) * 0.5

#lstm architecture
class lstm(object):
    def __init__(self,vocab_size):
        self.input = T.matrix()
        self.Wi = theano.shared(self.ortho_weight(128)[:100,:],borrow=True)
        self.Wf = theano.shared(self.ortho_weight(128)[:100,:],borrow=True)
        self.Wc = theano.shared(self.ortho_weight(128)[:100,:],borrow=True)
        self.Wo = theano.shared(self.ortho_weight(128)[:100,:],borrow=True)
        self.Ui = theano.shared(self.ortho_weight(128),borrow=True)
        self.Uf = theano.shared(self.ortho_weight(128),borrow=True)
        self.Uc = theano.shared(self.ortho_weight(128),borrow=True)
        self.Uo = theano.shared(self.ortho_weight(128),borrow=True)
        self.bi = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.bf = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.bc = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.bo = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.C0 = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.h0 = theano.shared(np.asarray(np.zeros(128),dtype=theano.config.floatX),borrow=True)
        self.W2 = theano.shared(np.random.randn(128,vocab_size).astype(theano.config.floatX)*np.sqrt(1.5/vocab_size),borrow=True)
        self.b2 = theano.shared(np.asarray(np.zeros(vocab_size),dtype=theano.config.floatX),borrow=True)
        self.target = T.matrix()
        
        self.params = [self.Wi,self.Wf,self.Wc,self.Wo,self.Ui,self.Uf,self.Uc,self.Uo,self.bi,self.bf,self.bc,self.bo,self.h0,self.C0,self.W2,self.b2]
        [self.c,self.h_output],_ = theano.scan(fn=self.step,sequences=self.input,outputs_info=[self.C0,self.h0],non_sequences=self.params[:-4])
        self.output =  T.nnet.softmax(T.dot(self.h_output,self.W2)+self.b2)[20:,:]
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.updates = self.adam(self.cost,self.params)
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

#initialize lstm
print 'initializing lstm'
NN = lstm(len(w2v.vocab))

#window of words to train over per step
batch = 120

#train lstm
for i in range(1000000):
    
    #pick random start point in text
    start = np.random.randint(20,len(dataset)-batch-1)
    
    #intialize word2vec arrays
    X_train = np.empty((batch,100))
    y_train = np.zeros((batch-20,len(w2v.vocab)))
    
    #fill in arrays with word2vec embeddings
    for j in range(batch):
        X_train[j,:] = vocab[word2id[dataset[start-20+j]],:]
        if j < batch-20:
            y_train[j,word2id[dataset[start+1+j]]] = 1
    
    cost = NN.train(X_train,y_train)
    sys.stdout.write("step %i training error: %f   \r" % (i+1, cost))
    sys.stdout.flush()
    
    #try generating text
    if (i+1) % 1000 == 0:
        
        #variable for generated string
        string = 'step %i prediction: ' % (i+1)
        
        #use random section of text to initialize
        start = np.random.randint(0,len(dataset)-21)
        X_test = np.empty((21,100))
        
        #fill in array with word2vec embeddings
        for j in range(21):
            string += dataset[start+j] + ' '
            X_test[j,:] = vocab[word2id[dataset[start+j]],:]
        
        #predict next word
        out = NN.predict(X_test)[0]
        
        #predict next 40 words
        for j in range(40):
            next = id2word[np.argmax(out)]
            string += next + ' '
            
            #predict next word
            next = vocab[word2id[next],:]
            X_test = np.vstack((X_test[1:,:],next))
            out = NN.predict(X_test)[0]
        
        print string