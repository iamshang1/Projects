import os
import numpy as np
import tensorflow as tf
import sys
import time

class text_cnn(object):

    def __init__(self,embedding_matrix,num_classes,max_words,num_filters=100,dropout_keep=0.5):
        '''
        word-level convolutional neural network for text classification
        
        parameters:
          - embedding_matrix: numpy array
            numpy array of word embeddings
            each row should represent a word embedding
            NOTE: the word index 0 is dropped, so the first row is ignored
          - num_classes: int
            number of output classes
          - max_sents: int
            maximum number of sentences per document
          - max_words: int
            maximum number of words per sentence
          - num_filters: int (default: 100)
            number of convolutional filters to use for each of the 3 convolution filters
          - dropout_keep: float (default: 0.5)
            dropout keep rate for final document representation
           
        methods:
          - train(,data,labels,validation_data,epochs=30,savebest=False,filepath=None)
            train network on given data
          - predict(data)
            return the one-hot-encoded predicted labels for given data
          - score(data,labels)
            return the accuracy of predicted labels on given data
          - save(filepath)
            save the model weights to a file
          - load(filepath)
            load model weights from a file
        '''

        self.vocab = embedding_matrix
        self.embedding_size = embedding_matrix.shape[1]
        self.embeddings = embedding_matrix.astype(np.float32)
        self.mw = max_words
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
                
        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_words])
        self.num_words = tf.reduce_sum(tf.sign(self.doc_input))
        self.doc_input_reduced = tf.expand_dims(self.doc_input[:self.num_words],0)
        
        #word embeddings
        self.word_embeds = tf.gather(tf.get_variable('embeddings',initializer=
                           self.embeddings,dtype=tf.float32),self.doc_input_reduced)
        
        #word convolutions
        conv3 = tf.layers.conv1d(self.word_embeds,num_filters,3,padding='same',
                activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer())
        conv4 = tf.layers.conv1d(self.word_embeds,num_filters,4,padding='same',
                activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer())
        conv5 = tf.layers.conv1d(self.word_embeds,num_filters,5,padding='same',
                activation=tf.nn.relu,kernel_initializer=tf.orthogonal_initializer())
        pool3 = tf.reduce_max(conv3,1)
        pool4 = tf.reduce_max(conv4,1)
        pool5 = tf.reduce_max(conv5,1)
        
        #concatenate
        concat = tf.concat([pool3,pool4,pool5],1)
        self.doc_embed = tf.nn.dropout(concat,self.dropout)
        
        #classification functions
        self.output = tf.layers.dense(self.doc_embed,num_classes,kernel_initializer=tf.orthogonal_initializer())
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(0.00002,0.9,0.99).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros(self.mw)
            for i,word in enumerate(inputval):
                retval[i] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
     
    def train(self,data,labels,epochs=5,validation_data=None,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - epochs: int (default: 30)
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
        
        print('training network on %i documents, validating on %i documents' \
              % (len(data), validation_size))
        
        #track best model for saving
        prevbest = 0    
        for i in range(epochs):
            correct = 0.
            start = time.time()
            
            #train
            for doc in range(len(data)):
                inputval = self._list_to_numpy(data[doc])
                feed_dict = {self.doc_input:inputval,self.labels:labels[doc],self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,len(data),cost))
                sys.stdout.flush()
                
                #checkpoint every 50000 iterations
                if (doc+1) % 50000 == 0:
                    print("\ntraining time: %.2f" % (time.time()-start))
                    score = self.score(validation_data[0],validation_data[1])
                    print("iteration %i validation accuracy: %.4f%%" % (doc+1, score*100))
                    
                    #reset timer
                    start = time.time()
                        
                    #save if performance better than previous best
                    if savebest and score >= prevbest:
                        prevbest = score
                        self.save(filepath)
                    
            print()
            trainscore = correct/len(data)
            print("epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100))

    def predict(self,data):
        '''
        return the one-hot-encoded predicted labels for given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
        
        outputs:
            numpy array of one-hot-encoded predicted labels for input data
        '''
        labels = []
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            prob = np.squeeze(prob,0)
            one_hot = np.zeros_like(prob)
            one_hot[np.argmax(prob)] = 1
            labels.append(one_hot)
        
        labels = np.array(labels)
        return labels
        
    def score(self,data,labels):
        '''
        return the accuracy of predicted labels on given data

        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
        
        outputs:
            float representing accuracy of predicted labels on given data
        '''        
        correct = 0.
        for doc in range(len(data)):
            inputval = self._list_to_numpy(data[doc])
            feed_dict = {self.doc_input:inputval,self.dropout:1.0}
            prob = self.sess.run(self.prediction,feed_dict=feed_dict)
            if np.argmax(prob) == np.argmax(labels[doc]):
                correct +=1

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

    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split
    import pickle
    import os

    #load saved files
    print("loading data")
    vocab = np.load('data/yelp16_embeddings.npy')
    with open('data/yelp16_data.pkl', 'rb') as f:
        data = pickle.load(f)

    num_docs = len(data)

    #convert data to numpy arrays
    print "converting data to arrays"
    max_words = 0
    docs = []
    labels = []
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
        sys.stdout.flush()
        doc = data[i]['idx']
        doc = [item for sublist in doc for item in sublist]
        docs.append(doc)
        labels.append(data[i]['label'])
        if len(doc) > max_words:
            max_words = len(doc)
    del data
    print

    #label encoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = len(le.classes_)
    lb = LabelBinarizer()
    lb.fit(y)
    del labels

    #test train split
    X_train,X_test,y_train,y_test = train_test_split(docs,y,test_size=0.1,
                                    random_state=1234,stratify=y)
                                    
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.12,
                                      random_state=1234,stratify=y_train)

    y_train = lb.transform(y_train)
    y_valid = lb.transform(y_valid)
    y_test = lb.transform(y_test)

    #create directory for saved model
    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')

    #train nn
    print("building text cnn")
    nn = text_cnn(vocab,classes,max_words)
    nn.train(X_train,y_train,epochs=3,validation_data=(X_valid,y_valid),
             savebest=True,filepath='savedmodels/cnn_yelp16.ckpt')
    
    #load best nn and test
    nn.load('savedmodels/cnn_yelp16.ckpt')
    score = nn.score(X_test,y_test)
    print("final test accuracy: %.4f%%" % (score*100))
