import pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys

#convolutional network for text classification
class text_cnn(object):
    '''
    convolutional network for text classification
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - feature_maps: int (default: 100)
        number of output channels for each convolution
      - dropout_rate: float (default: 0.5)
        dropout rate for embeddings
       
    methods:
      - train(data,labels,epochs=30,savebest=False,filepath=None)
        train network on given data
      - predict(data)
        return the one-hot-encoded predicted labels for given data
      - predict_proba(data)
        return the probability of each class for given data
      - score(data,labels)
        return the accuracy of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    '''
    def __init__(self,embedding_matrix,num_classes,feature_maps=100,dropout_rate=0.5):
    
        embedding_size = embedding_matrix.shape[1]
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(1234)
        self.dropout_on = T.iscalar()
        
        #word word embeddings
        self.embeddings = theano.shared(embedding_matrix.astype(theano.config.floatX))
        
        #input layer
        self.input = T.ivector()

        #remove all 0 index values from line
        self.masked_word_ids = self.input.nonzero_values()
        self.embedded_words = T.take(self.embeddings, self.masked_word_ids, axis=0)
        
        #add first (batch) dimension and second (channels) dimension
        self.embedded_words = self.embedded_words.dimshuffle('x','x',0,1)
        
        #pad sentence with ones to make sure its larger than largest convolution window
        self.ones = theano.shared(np.zeros((1,1,2,embedding_size), dtype=theano.config.floatX))
        self.embedded_words = T.concatenate([self.ones,self.embedded_words,self.ones],axis=2)

        #word convolution layers
        self.conv3W = theano.shared(self._conv_ortho_weight(feature_maps,1,3,embedding_size))
        self.conv4W = theano.shared(self._conv_ortho_weight(feature_maps,1,4,embedding_size))
        self.conv5W = theano.shared(self._conv_ortho_weight(feature_maps,1,5,embedding_size))
        self.conv3b = theano.shared(np.zeros((feature_maps,), dtype=theano.config.floatX))
        self.conv4b = theano.shared(np.zeros((feature_maps,), dtype=theano.config.floatX))
        self.conv5b = theano.shared(np.zeros((feature_maps,), dtype=theano.config.floatX))
        
        #word convolution ops
        self.conv3op = conv2d(input=self.embedded_words, filters=self.conv3W, border_mode='valid')
        self.conv4op = conv2d(input=self.embedded_words, filters=self.conv4W, border_mode='valid')
        self.conv5op = conv2d(input=self.embedded_words, filters=self.conv5W, border_mode='valid')
        self.conv3nl = T.nnet.elu(self.conv3op + self.conv3b.dimshuffle('x', 0, 'x', 'x'))
        self.conv4nl = T.nnet.elu(self.conv4op + self.conv4b.dimshuffle('x', 0, 'x', 'x'))
        self.conv5nl = T.nnet.elu(self.conv5op + self.conv5b.dimshuffle('x', 0, 'x', 'x'))
        
        #word maxpool operation
        self.conv3max = T.max(self.conv3nl,axis=2).flatten()
        self.conv4max = T.max(self.conv4nl,axis=2).flatten()
        self.conv5max = T.max(self.conv5nl,axis=2).flatten()
                              
        #concatenate maxpools from each word window
        self.flattened = T.concatenate([self.conv3max,self.conv4max,self.conv5max])
        self.flattened_drop = self._dropout(self.flattened,(feature_maps*3,),self.dropout_on)
        
        #softmax function over classes
        self.W_softmax = theano.shared(self._ortho_weight(feature_maps*3,num_classes))
        self.b_softmax = theano.shared(np.asarray(np.zeros(num_classes),dtype=theano.config.floatX))
        self.output = T.nnet.softmax(T.dot(self.flattened_drop,self.W_softmax)+self.b_softmax).flatten()

        #training, predict, and update functions
        self.params = [self.conv3W,self.conv4W,self.conv5W,self.conv3b,self.conv4b,self.conv5b,\
                       self.W_softmax,self.b_softmax,self.embeddings]
        self.target = T.vector()
        
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.updates = self._adam(self.cost,self.params)
        self.train_f = theano.function([self.input,self.target,self.dropout_on],[self.output,self.cost],
                                       updates=self.updates,allow_input_downcast=True)
        self.predict_f = theano.function([self.input,self.dropout_on],self.output,allow_input_downcast=True)

    def _dropout_train(self,matrix,shape):
        '''
        dropout operation for training
        '''
        srng = RandomStreams(self.rng.randint(999999))
        return T.switch(srng.binomial(size=shape,p=1-self.dropout_rate),matrix,0)

    def _dropout_predict(self,matrix):
        '''
        dropout operation for prediction
        '''
        return (1-self.dropout_rate)*matrix
    
    def _dropout(self,matrix,shape,training):
        '''
        dropout operation for train and predict
        '''
        result = theano.ifelse.ifelse(theano.tensor.eq(training, 1),
                 self._dropout_train(matrix,shape), self._dropout_predict(matrix))
        return result

    def _ortho_weight(self,fan_in,fan_out):
        '''
        generate orthogonal weight matrix
        '''
        bound = np.sqrt(2./(fan_in+fan_out))
        W = np.random.randn(fan_in,fan_out)*bound
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u
        else:
            W = v
        return W.astype(theano.config.floatX)
        
    def _conv_ortho_weight(self,chan_out,chan_in,filter_h,filter_w):
        '''
        generate convolutional orthogonal weight matrix
        '''
        bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
        W = np.random.random((chan_out, chan_in * filter_h * filter_w))
        u, s, v = np.linalg.svd(W,full_matrices=False)
        if u.shape[0] != u.shape[1]:
            W = u.reshape((chan_out, chan_in, filter_h, filter_w))
        else:
            W = v.reshape((chan_out, chan_in, filter_h, filter_w))
        return W.astype(theano.config.floatX)

    def _adam(self, cost, params, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
        '''
        adam gradient descent updates
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
        
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,
              filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 10)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True
        
        outputs:
            None
        '''
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print 'training network on %i documents, validating on %i documents' \
              % (len(data), validation_size)
              
        prevbest = 0
        for i in range(epochs):
            correct = 0.
            ctr = 0
            for doc in range(len(data)):
                pred,cost = self.train_f(data[doc],labels[doc],1)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,len(data),cost))
                sys.stdout.flush()
            print ""
            trainscore = correct/len(data)
            print "epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100)
            if validation_data:
                score = self.score(validation_data[0],validation_data[1])
                print "epoch %i validation accuracy: %.4f%%" % (i+1, score*100)
            else:
                score = self.score(data,labels)
                print "epoch %i validation accuracy: %.4f%%" % (i+1, score*100)
            if savebest and score >= prevbest:
                prevbest = score
                self.save(filepath)
        
    def predict(self,data):
        '''
        return the one-hot-encoded predicted labels for given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
        
        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        '''        
        labels = []
        for doc in range(len(data)):
            prob = self.predict_f(data[doc],0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)
        
        labels = np.array(labels)
        return labels
        
    def predict_proba(self,data):
        '''
        return the probability of each class for given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
        
        outputs:
            numpy array of probability of each class for input data
        '''
        probs = []
        for doc in range(len(data)):
            prob = self.predict_f(data[doc],0)
            probs.append(prob)
        
        probs = np.array(probs)
        return probs
        
    def score(self,data,labels):
        '''
        return the accuracy of predicted labels on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
        
        outputs:
            float representing accuracy of predicted labels on given data
        '''
        correct = 0.
        for doc in range(len(data)):
            prob = self.predict_f(data[doc],0)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct += 1
        
        accuracy = correct/len(labels)
        return accuracy
        
    def save(self,filename):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        params_dic = {}
        for idx,param in enumerate(self.params):
            params_dic[idx] = param
            
        with open(filename, 'wb') as f:
            pickle.dump(params_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load(self,filename):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        with open(filename, 'rb') as f:
            params_dic = pickle.load(f)

        for idx,param in enumerate(self.params):
            self.params[idx] = params_dic[idx]
            

if __name__ == "__main__":

    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split

    #load saved files
    print "loading data"
    vocab = np.load('embeddings.npy')
    with open('data.p', 'rb') as f:
        data = pickle.load(f)

    num_docs = len(data)

    #convert data to numpy arrays
    print "converting data to arrays"
    docs = []
    labels = []
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
        sys.stdout.flush()
        docs.append(data[i]['idx'])
        labels.append(data[i]['label'])
    del data
    print

    #label encoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = len(le.classes_)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y)
    del labels

    #flatten docs
    print "flattening docs"
    maxlen = 0
    flattened_docs = []
    for i,doc in enumerate(docs):
        sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
        sys.stdout.flush()
        flattened = [idx for line in doc for idx in line]
        if len(flattened) > maxlen:
            maxlen = len(flattened)
        flattened_docs.append(flattened)
    del docs
    print

    #test train split
    X_train,X_test,y_train,y_test = train_test_split(flattened_docs,y_bin,
                                    test_size=0.1,random_state=1234,stratify=y)
    
    #train nn
    print "building text cnn"
    nn = text_cnn(vocab,classes)
    nn.train(X_train,y_train,epochs=5,validation_data=(X_test,y_test),
             savebest=True,filepath='cnn.p')
    
    #load best nn
    nn.load('cnn.p')
    acc = nn.score(X_test,y_test)
    y_pred = np.argmax(nn.predict(X_test),1)
    print "CNN - test set accuracy: %.4f" % (acc*100)
