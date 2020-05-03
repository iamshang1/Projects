import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
import sys
import time
from sklearn.metrics import f1_score
import random

class han(object):
    '''
    hierarchical attention network by yang et. al.
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num_classes: int
        number of output classes
      - max_sents: int
        maximum number of sentences/lines per document
      - max_words: int
        maximum number of words per sentence/line
      - rnn_type: string (default: "gru")
        rnn cells to use, can be "gru" or "lstm"
      - rnn_units: int (default: 50)
        number of rnn units to use for embedding layers
      - attention_size: int (default: 200)
        number of dimensions to use for attention hidden layer
      - lr: float (default: 0.0001)
        learning rate for adam optimizer
       
    methods:
      - train(data,labels,batch_size=64,epochs=30,patience=5,
              validation_data,savebest=False,filepath=None)
        train network on given data
      - predict(data)
        return the predicted labels for given data
      - score(data,labels)
        return the micro and macro f-scores of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    '''
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,rnn_type="gru",
                 rnn_units=200,attention_size=300,lr=0.0001):
        
        self.rnn_units = rnn_units
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.ms = max_sents
        self.mw = max_words
        self.attention_size = attention_size
        
        #doc input
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
        doc_embed = tf.map_fn(self._han_step,self.doc_input,dtype=tf.float32)

        #classification functions
        output = tf.layers.dense(doc_embed,num_classes,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.prediction = tf.nn.softmax(output)

        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.int32,shape=[None])
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output,labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _han_step(self,doc):

        words_per_line = tf.math.count_nonzero(doc,1)
        num_lines = tf.math.count_nonzero(words_per_line)
        max_words_ = tf.reduce_max(words_per_line)
        doc_input_reduced = doc[:num_lines,:max_words_]
        num_words = words_per_line[:num_lines]

        #word embeddings
        word_embeds = tf.gather(tf.get_variable('embeddings',
                      initializer=self.embedding_matrix,dtype=tf.float32),
                      doc_input_reduced)

        #word rnn layer
        with tf.variable_scope('word',initializer=tf.contrib.layers.xavier_initializer()):
            [word_outputs_fw,word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    word_embeds,sequence_length=num_words,dtype=tf.float32)
        word_outputs = tf.concat((word_outputs_fw,word_outputs_bw),2)
 
        #word attention
        seq_mask = tf.reshape(tf.sequence_mask(num_words,max_words_),[-1])
        word_u = tf.layers.dense(tf.reshape(word_outputs,[-1,self.rnn_units*2]),
                 self.attention_size,tf.nn.tanh,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        word_exps = tf.layers.dense(word_u,1,tf.exp,False,
                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        word_exps = tf.where(seq_mask,word_exps,tf.ones_like(word_exps)*0.000000001)
        word_alpha = tf.reshape(word_exps,[-1,max_words_,1])
        word_alpha /= tf.reshape(tf.reduce_sum(word_alpha,1),[-1,1,1])
        sent_embeds = tf.reduce_sum(word_outputs*word_alpha,1)
        sent_embeds = tf.expand_dims(sent_embeds,0)

        #sentence rnn layer
        with tf.variable_scope('sentence',initializer=tf.contrib.layers.xavier_initializer()):
            [sent_outputs_fw,sent_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    sent_embeds,sequence_length=tf.expand_dims(num_lines,0),dtype=tf.float32)
        sent_outputs = tf.concat((sent_outputs_fw,sent_outputs_bw),2)
        sent_outputs = tf.squeeze(sent_outputs,[0])

        #sentence attention
        u = tf.layers.dense(sent_outputs,self.attention_size,tf.nn.tanh,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        exps = tf.layers.dense(u,1,tf.exp,False,
               kernel_initializer=tf.contrib.layers.xavier_initializer())
        atten = exps/tf.reduce_sum(exps)
        doc_embed = tf.transpose(tf.matmul(tf.transpose(sent_outputs),atten))

        return tf.squeeze(doc_embed,[0])
    
    def train(self,data,labels,batch_size=64,epochs=30,patience=5,
              validation_data=None,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            1d numpy array of labels for given data
          - batch size: int (default: 64)
            batch size to use for training
          - epochs: int (default: 30)
            number of epochs to train for
          - patience: int (default: 5)
            training stops after no improvement in validation score
            for this number of epochs
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

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        #track best model for saving
        prevbest = 0
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            xy = list(zip(data,labels))
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = list(data)
            labels = np.array(labels)

            y_pred = []
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.doc_input:data[start:stop],
                             self.labels:labels[start:stop]}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],
                              feed_dict=feed_dict)
                              
                #track correct predictions
                y_pred.append(np.argmax(pred,1))
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop,len(data),cost))
                sys.stdout.flush()
                
            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            y_pred = np.concatenate(y_pred,0)
            micro = f1_score(labels,y_pred,average='micro')
            macro = f1_score(labels,y_pred,average='macro')
            print("epoch %i training micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            micro,macro = self.score(validation_data[0],validation_data[1],
                                     batch_size=batch_size)
            print("epoch %i validation micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            #save if performance better than previous best
            if micro >= prevbest:
                prevbest = micro
                pat_count = 0
                if savebest:
                    self.save(filepath)
            else:
                pat_count += 1
                if pat_count >= patience:
                    break

            #reset timer
            start_time = time.time()

    def predict(self,data,batch_size=64):
        '''
        return the predicted labels for given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - batch size: int (default: 64)
            batch size to use during inference
            
        outputs:
            1d numpy array of predicted labels for input data
        '''
        
        y_pred = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input:data[start:stop]}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            y_pred.append(np.argmax(prob,1))

            sys.stdout.write("processed %i of %i records        \r" \
                             % (stop,len(data)))
            sys.stdout.flush()

        print()
        y_pred = np.concatenate(y_pred,0)
        return y_pred

    def score(self,data,labels,batch_size=64):
        '''
        return the micro and macro f-score of predicted labels on given data

        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            1d numpy array of labels for given data
          - batch size: int (default: 64)
            batch size to use during inference
        
        outputs:
            tuple of floats (micro,macro) representing micro and macro f-score
            of predicted labels on given data
        '''  
        
        y_pred = self.predict(data,batch_size)
        micro = f1_score(labels,y_pred,average='micro')
        macro = f1_score(labels,y_pred,average='macro')
        return micro,macro

    def save(self,filename):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    '''
    dummy test data
    '''
    
    #params
    batch_size = 64
    lr = 0.0001
    epochs = 5
    train_samples = 10000
    test_samples = 10000
    vocab_size = 750
    max_lines = 50
    max_words = 10
    num_classes = 10
    embedding_size = 100
    rnn_units = 50
    attention_size = 100
    
    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(1,vocab_size,
        (train_samples+test_samples,max_lines,max_words))

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = np.random.randint(0,num_classes,train_samples)
    y_test = np.random.randint(0,num_classes,test_samples)

    #train model
    model = han(vocab,num_classes,max_lines,max_words,
                rnn_units=rnn_units,attention_size=attention_size,lr=lr)
    model.train(X_train,y_train,batch_size,epochs,
                validation_data=(X_test,y_test))