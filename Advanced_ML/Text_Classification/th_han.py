'''
hierarchical attention network for document classification
https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
'''

import pickle
import numpy as np
import theano
import theano.tensor as T
import sys

#heirarchical attention network
class hierarchical_attention_network(object):
    '''
    hierarchical attention network for document classification
    https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - lstm units: int (default: 50)
        number of lstm units to use for embedding layers
      - attention_context: int (default: 100)
        number of dimensions to use for attention context layer
       
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
    def __init__(self,embedding_matrix,num_classes,lstm_units=50,attention_context=100):
    
        embedding_size = embedding_matrix.shape[1]
    
        #word word embeddings
        self.embeddings = theano.shared(embedding_matrix.astype(theano.config.floatX))

        #word forward lstm layer
        self.word_forward_outputs_info, self.word_forward_non_sequences \
            = self._create_lstm_layer(embedding_size,lstm_units)
        
        #word backward lstm layer
        self.word_backward_outputs_info, self.word_backward_non_sequences \
            = self._create_lstm_layer(embedding_size,lstm_units)
        
        #word attention layer
        self.word_W = theano.shared(self._ortho_weight(2*lstm_units,attention_context))
        self.word_b = theano.shared(np.asarray(np.zeros(attention_context),dtype=theano.config.floatX))
        self.word_context = theano.shared(self._ortho_weight(attention_context,1))
        
        #doc input layer
        self.doc_input = T.imatrix()
        self.nonzeros = T.nonzero(T.sum(self.doc_input,1))
        self.masked_doc = self.doc_input[self.nonzeros]
        
        #get line embeddings
        self.line_embeddings,_ = theano.scan(fn=self._line_embedding_step,sequences=self.masked_doc)
        
        #line forward lstm layer
        self.line_forward_outputs_info, self.line_forward_non_sequences \
            = self._create_lstm_layer(lstm_units*2,lstm_units)
        [_,self.line_forward_out],_ = theano.scan(fn=self._lstm_step,sequences=self.line_embeddings,
                                      outputs_info=self.line_forward_outputs_info,
                                      non_sequences=self.line_forward_non_sequences)
        
        #line backward lstm layer
        self.line_backward_outputs_info, self.line_backward_non_sequences \
            = self._create_lstm_layer(lstm_units*2,lstm_units)
        self.line_embeddings_backward = self.line_embeddings[::-1]
        [_,self.line_backward_out],_ = theano.scan(fn=self._lstm_step,
                                       sequences=self.line_embeddings_backward,
                                       outputs_info=self.line_backward_outputs_info,
                                       non_sequences=self.line_backward_non_sequences)
        
        #concatenate forward and backward lstm embeddings at each step
        self.line_lstm_output = T.concatenate([self.line_forward_out,self.line_backward_out[::-1]],1)
        
        #line attention layer
        self.line_W = theano.shared(self._ortho_weight(2*lstm_units,attention_context))
        self.line_b = theano.shared(np.asarray(np.zeros(attention_context),dtype=theano.config.floatX))
        self.line_context = theano.shared(self._ortho_weight(attention_context,1))
        
        #line attention functions
        self.line_attention,_ = theano.scan(fn=self._attention_step,sequences=self.line_lstm_output,
                                  non_sequences=[self.line_W,self.line_b,self.line_context])
        self.line_importance = self.line_attention/T.sum(self.line_attention)
        
        #create doc embedding
        self.doc_embedding = T.dot(self.line_lstm_output.T,self.line_importance).flatten()
        
        #softmax function over classes
        self.W_softmax = theano.shared(self._ortho_weight(lstm_units*2,num_classes))
        self.b_softmax = theano.shared(np.asarray(np.zeros(num_classes),dtype=theano.config.floatX))
        self.output = T.nnet.softmax(T.dot(self.doc_embedding,self.W_softmax)+self.b_softmax).flatten()

        #training, predict, and update functions
        self.params = [self.embeddings,self.word_W,self.word_b,self.word_context,
                       self.line_W,self.line_b,self.line_context,self.W_softmax,self.b_softmax]\
                       + self.word_forward_outputs_info + self.word_forward_non_sequences \
                       + self.word_backward_outputs_info + self.word_backward_non_sequences \
                       + self.line_forward_outputs_info + self.line_forward_non_sequences \
                       + self.line_backward_outputs_info + self.line_backward_non_sequences
        self.target = T.vector()
        self.cost = T.nnet.categorical_crossentropy(self.output,self.target).mean()
        self.updates = self._adam(self.cost,self.params)
        self.train_f = theano.function([self.doc_input,self.target],[self.output,self.cost],
                                       updates=self.updates,allow_input_downcast=True)
        self.predict_f = theano.function([self.doc_input],self.output,allow_input_downcast=True)
    
    def _create_lstm_layer(self,input_size,output_size):
        '''
        create layer of lstm units
        '''
        Wi = theano.shared(self._ortho_weight(input_size,output_size))
        Wf = theano.shared(self._ortho_weight(input_size,output_size))
        Wc = theano.shared(self._ortho_weight(input_size,output_size))
        Wo = theano.shared(self._ortho_weight(input_size,output_size))
        Ui = theano.shared(self._ortho_weight(output_size,output_size))
        Uf = theano.shared(self._ortho_weight(output_size,output_size))
        Uc = theano.shared(self._ortho_weight(output_size,output_size))
        Uo = theano.shared(self._ortho_weight(output_size,output_size))
        bi = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        bf = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        bc = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        bo = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        C0 = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        h0 = theano.shared(np.asarray(np.zeros(output_size),dtype=theano.config.floatX))
        outputs_info = [C0,h0]
        non_sequences = [Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo]
        return outputs_info, non_sequences

    def _lstm_step(self,embeddings,h0,C0,Wi,Wf,Wc,Wo,Ui,Uf,Uc,Uo,bi,bf,bc,bo):
        '''
        lstm memory cell functions
        '''
        i = T.nnet.sigmoid(T.dot(embeddings,Wi)+T.dot(h0,Ui)+bi)
        cand = T.tanh(T.dot(embeddings,Wc)+T.dot(h0,Uc)+bc)
        f = T.nnet.sigmoid(T.dot(embeddings,Wf)+T.dot(h0,Uf)+bf)
        c = cand*i+C0*f
        o = T.nnet.sigmoid(T.dot(embeddings,Wo)+T.dot(h0,Uo)+bo)
        h = o*T.tanh(c)
        return c,h
    
    def _attention_step(self,lstm_output,W,b,context):
        '''
        attention layer functions
        '''
        u = T.tanh(T.dot(lstm_output,W) + b)
        a = T.exp(T.dot(u,context))
        return a

    def _line_embedding_step(self,line):
        '''
        get line embeddings
        '''
        #remove all 0 index values from line
        masked_word_ids = line.nonzero_values()
        embedded_words = T.take(self.embeddings, masked_word_ids, axis=0)

        #word forward lstm
        [_,word_forward_out],_ = theano.scan(fn=self._lstm_step,sequences=embedded_words,
                                 outputs_info=self.word_forward_outputs_info,
                                 non_sequences=self.word_forward_non_sequences)
        
        #word backward lstm
        embedded_words_backward = embedded_words[::-1]
        [_,word_backward_out],_ = theano.scan(fn=self._lstm_step,
                                  sequences=embedded_words_backward,
                                  outputs_info=self.word_backward_outputs_info,
                                  non_sequences=self.word_backward_non_sequences)
                              
        #concatenate forward and backward lstm embeddings at each step
        word_lstm_output = T.concatenate([word_forward_out,word_backward_out[::-1]],1)
        
        #word attention
        word_attention,_ = theano.scan(fn=self._attention_step,sequences=word_lstm_output,
                           non_sequences=[self.word_W,self.word_b,self.word_context])
        word_importance = word_attention/T.sum(word_attention)
        
        #create line embedding
        line_embedding = T.dot(word_lstm_output.T,word_importance).flatten()
        return line_embedding

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
        
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            maxlen = len(max(inputval,key=len))
            retval = np.zeros((len(inputval),maxlen))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.array:
            return inputval
        else:
            raise Exception("invalid input type")
        
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,
              filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
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
                inputval = self._list_to_numpy(data[doc])
                pred,cost = self.train_f(inputval,labels[doc])
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
            inputval = self._list_to_numpy(data[doc])
            prob = self.predict_f(inputval)
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
            inputval = self._list_to_numpy(data[doc])
            prob = self.predict_f(inputval)
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
            inputval = self._list_to_numpy(data[doc])
            prob = self.predict_f(inputval)
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

    #test train split
    X_train,X_test,y_train,y_test = train_test_split(docs,y_bin,test_size=0.1,
                                    random_state=1234,stratify=y)

    #train nn
    print "building hierarchical attention network"
    nn = hierarchical_attention_network(vocab,classes)
    nn.train(X_train,y_train,epochs=5,validation_data=(X_test,y_test),
             savebest=True,filepath='han.p')
    
    #load best nn
    nn.load('han.p')
    acc = nn.score(X_test,y_test)
    y_pred = np.argmax(nn.predict(X_test),1)
    print "HAN - test set accuracy: %.4f" % (acc*100)
