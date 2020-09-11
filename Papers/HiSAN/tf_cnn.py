import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics import f1_score
import random

class cnn(object):
    '''
    text cnn based off yoon kim cnn
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num_classes: int
        number of output classes
      - max_words: int
        maximum number of words per document
      - num_filters: int (default: 300)
        number of filters for each of the three convolutional layers
      - dropout_keep: float (default: 0.9)
        dropout keep rate after maxpool
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
    def __init__(self,embedding_matrix,num_classes,max_words,
                 num_filters=300,dropout_keep=0.5,lr=0.0001):
        
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.mw = max_words
        self.initializer = tf.contrib.layers.xavier_initializer()

        #doc input and embeddings
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_words])
        embeddings = tf.get_variable('embeddings',initializer=
                     embedding_matrix.astype(np.float32),dtype=tf.float32)
        word_embeds = tf.gather(embeddings,self.doc_input)

        #word convolutions
        conv3 = tf.layers.conv1d(word_embeds,num_filters,3,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        conv4 = tf.layers.conv1d(word_embeds,num_filters,4,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        conv5 = tf.layers.conv1d(word_embeds,num_filters,5,padding='same',
                activation=tf.nn.relu,kernel_initializer=self.initializer)
        pool3 = tf.reduce_max(conv3,1)
        pool4 = tf.reduce_max(conv4,1)
        pool5 = tf.reduce_max(conv5,1)

        #concatenate
        concat = tf.concat([pool3,pool4,pool5],1)
        doc_embed = tf.nn.dropout(concat,self.dropout)

        #classification functions
        output = tf.layers.dense(doc_embed,num_classes,
                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.prediction = tf.nn.softmax(output)

        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.int32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output,labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,data,labels,batch_size=64,epochs=30,patience=5,
              validation_data=None,savebest=False,filepath=None):
         '''
        train network on given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
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
            labels = list(labels)

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
                             self.labels:labels[start:stop],
                             self.dropout:self.dropout_keep}
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

            micro,macro = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
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

    def predict(self,data,batch_size=16):
        '''
        return the predicted labels for given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
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

            feed_dict = {self.doc_input:data[start:stop],self.dropout:1.0}
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
            2d numpy array (doc x word ids) of input data
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
    max_words = 500
    num_classes = 10
    embedding_size = 100
    num_filters = 100
    
    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    vocab[0,:] = 0
    X = np.random.randint(1,vocab_size,
        (train_samples+test_samples,max_words))

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_train = np.random.randint(0,num_classes,train_samples)
    y_test = np.random.randint(0,num_classes,test_samples)

    #train model
    model = cnn(vocab,num_classes,max_words,num_filters,lr=lr)
    model.train(X_train,y_train,batch_size,epochs,
                validation_data=(X_test,y_test))