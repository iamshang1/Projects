import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import layer_norm
import sys
import time

class hcan(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,activation=tf.nn.elu):
        '''
        hierarchical convolutional attention network for text classification
        
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
          - attention_heads: int (default: 8)
            number of attention heads to use in multihead attention
          - attention_size: int (default: 512)
            dimension size of output embeddings from attention 
          - dropout_keep: float (default: 0.9)
            dropout keep rate for embeddings and attention softmax
          - activation: tensorflow activation function (default: tf.nn.elu)
            activation function to use for convolutional feature extraction
           
        methods:
          - train(data,labels,validation_data,epochs=30,savebest=False,filepath=None)
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
    
        self.attention_heads = attention_heads
        self.attention_size = attention_size
        self.embedding_size = embedding_matrix.shape[1]
        self.embeddings = embedding_matrix.astype(np.float32)
        self.ms = max_sents
        self.mw = max_words
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
                
        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        self.words_per_line = tf.reduce_sum(tf.sign(self.doc_input),1)
        self.max_lines = tf.reduce_sum(tf.sign(self.words_per_line))
        self.max_words = tf.reduce_max(self.words_per_line)
        self.doc_input_reduced = self.doc_input[:self.max_lines,:self.max_words]
        self.num_words = self.words_per_line[:self.max_lines]
        
        #word embeddings
        self.word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embeddings,
                           dtype=tf.float32),self.doc_input_reduced)
        positions = tf.expand_dims(tf.range(self.max_words),0)
        word_pos = tf.gather(tf.get_variable('word_pos',shape=(self.mw,self.embedding_size), 
                   dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)),positions)              
        self.word_embeds = tf.nn.dropout(self.word_embeds + word_pos,self.dropout)
        
        #masks to eliminate padding
        mask_base = tf.cast(tf.sequence_mask(self.num_words,self.max_words),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,self.max_words])
        
        #word self attention 1
        Q1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V1 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        
        Q1 = tf.where(tf.equal(mask,0),tf.zeros_like(Q1),Q1)
        K1 = tf.where(tf.equal(mask,0),tf.zeros_like(K1),K1)
        V1 = tf.where(tf.equal(mask,0),tf.zeros_like(V1),V1)
        
        Q1_ = tf.concat(tf.split(Q1,self.attention_heads,axis=2),axis=0)
        K1_ = tf.concat(tf.split(K1,self.attention_heads,axis=2),axis=0)
        V1_ = tf.concat(tf.split(V1,self.attention_heads,axis=2),axis=0)
        
        outputs1 = tf.matmul(Q1_,tf.transpose(K1_,[0, 2, 1]))
        outputs1 = outputs1/(K1_.get_shape().as_list()[-1]**0.5)
        outputs1 = tf.where(tf.equal(outputs1,0),tf.ones_like(outputs1)*-1000,outputs1)
        outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),self.dropout)
        outputs1 = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs1),outputs1)
        outputs1 = tf.matmul(outputs1,V1_)
        outputs1 = tf.concat(tf.split(outputs1,self.attention_heads,axis=0),axis=2)
        outputs1 = tf.where(tf.equal(mask,0),tf.zeros_like(outputs1),outputs1)
        
        #word self attention 2
        Q2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V2 = tf.layers.conv1d(self.word_embeds,self.attention_size,3,padding='same',
            activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer())
        
        Q2 = tf.where(tf.equal(mask,0),tf.zeros_like(Q2),Q2)
        K2 = tf.where(tf.equal(mask,0),tf.zeros_like(K2),K2)
        V2 = tf.where(tf.equal(mask,0),tf.zeros_like(V2),V2)
        
        Q2_ = tf.concat(tf.split(Q2,self.attention_heads,axis=2),axis=0)
        K2_ = tf.concat(tf.split(K2,self.attention_heads,axis=2),axis=0)
        V2_ = tf.concat(tf.split(V2,self.attention_heads,axis=2),axis=0)
        
        outputs2 = tf.matmul(Q2_,tf.transpose(K2_,[0, 2, 1]))
        outputs2 = outputs2/(K2_.get_shape().as_list()[-1]**0.5)
        outputs2 = tf.where(tf.equal(outputs2,0),tf.ones_like(outputs2)*-1000,outputs2)
        outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),self.dropout)
        outputs2 = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs2),outputs2)
        outputs2 = tf.matmul(outputs2,V2_)
        outputs2 = tf.concat(tf.split(outputs2,self.attention_heads,axis=0),axis=2)
        outputs2 = tf.where(tf.equal(mask,0),tf.zeros_like(outputs2),outputs2)
        
        outputs = tf.multiply(outputs1,outputs2)
        outputs = layer_norm(outputs)
        
        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        K = tf.layers.conv1d(outputs,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
          
        Q = tf.tile(Q,[self.max_lines,1,1])
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)
        
        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        
        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        self.sent_embeds = tf.transpose(outputs,[1, 0, 2])
            
        #sentence positional embeddings
        positions = tf.expand_dims(tf.range(self.max_lines),0)
        sent_pos = tf.gather(tf.get_variable('sent_pos',shape=(self.ms,self.attention_size), 
                   dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)),positions)
        self.sent_embeds = tf.nn.dropout(self.sent_embeds + sent_pos,self.dropout)
            
        #sentence self attention 1
        Q1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V1 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        
        Q1_ = tf.concat(tf.split(Q1,self.attention_heads,axis=2),axis=0)
        K1_ = tf.concat(tf.split(K1,self.attention_heads,axis=2),axis=0)
        V1_ = tf.concat(tf.split(V1,self.attention_heads,axis=2),axis=0)
        
        outputs1 = tf.matmul(Q1_,tf.transpose(K1_,[0, 2, 1]))
        outputs1 = outputs1/(K1_.get_shape().as_list()[-1]**0.5)
        outputs1 = tf.nn.dropout(tf.nn.softmax(outputs1),self.dropout)
        outputs1 = tf.matmul(outputs1,V1_)
        outputs1 = tf.concat(tf.split(outputs1,self.attention_heads,axis=0),axis=2)
        
        #sentence self attention 2
        Q2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        K2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
        V2 = tf.layers.conv1d(self.sent_embeds,self.attention_size,3,padding='same',
            activation=tf.nn.tanh,kernel_initializer=tf.orthogonal_initializer())
        
        Q2_ = tf.concat(tf.split(Q2,self.attention_heads,axis=2),axis=0)
        K2_ = tf.concat(tf.split(K2,self.attention_heads,axis=2),axis=0)
        V2_ = tf.concat(tf.split(V2,self.attention_heads,axis=2),axis=0)
        
        outputs2 = tf.matmul(Q2_,tf.transpose(K2_,[0, 2, 1]))
        outputs2 = outputs2/(K2_.get_shape().as_list()[-1]**0.5)
        outputs2 = tf.nn.dropout(tf.nn.softmax(outputs2),self.dropout)
        outputs2 = tf.matmul(outputs2,V2_)
        outputs2 = tf.concat(tf.split(outputs2,self.attention_heads,axis=0),axis=2)
        
        outputs = tf.multiply(outputs1,outputs2)
        outputs = layer_norm(outputs)
        
        #sentence target attention
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        K = tf.layers.conv1d(outputs,self.attention_size,3,padding='same',
            activation=activation,kernel_initializer=tf.orthogonal_initializer())
           
        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        
        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        self.doc_embed = tf.nn.dropout(tf.squeeze(outputs,[0]),self.dropout)
        
        #classification functions
        self.output = tf.layers.dense(self.doc_embed,num_classes,
                      kernel_initializer=tf.orthogonal_initializer())
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                    (logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(2e-5,0.9,0.99).minimize(self.loss)

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
            retval = np.zeros((self.ms,self.mw))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
     
    def train(self,data,labels,validation_data,epochs=5,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
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
        
        validation_size = len(validation_data[0])
        
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
                feed_dict = {self.doc_input:inputval,self.labels:labels[doc],
                            self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],
                              feed_dict=feed_dict)
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
            3d numpy array (doc x sentence x word ids) of input data
        
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
            3d numpy array (doc x sentence x word ids) of input data
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
    vocab = np.load('data/yelp_embeddings.npy')
    with open('data/yelp_data.pkl', 'rb') as f:
        data = pickle.load(f)

    num_docs = len(data)

    #convert data to numpy arrays
    print("converting data to arrays")
    max_sents = 0
    max_words = 0
    docs = []
    labels = []
    for i in range(num_docs):
        sys.stdout.write("processing record %i of %i       \r" % (i+1,num_docs))
        sys.stdout.flush()
        doc = data[i]['idx']
        docs.append(doc)
        labels.append(data[i]['label'])
        if len(doc) > max_sents:
            max_sents = len(doc)
        if len(max(doc,key=len)) > max_words:
            max_words = len(max(doc,key=len))
    del data
    print()

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
    print("building convolutional attention network")
    nn = hcan(vocab,classes,max_sents,max_words)
    nn.train(X_train,y_train,epochs=3,validation_data=(X_valid,y_valid),
             savebest=True,filepath='savedmodels/hcan_yelp16.ckpt')
    
    #load best nn and test
    nn.load('savedmodels/hcan_yelp16.ckpt')
    score = nn.score(X_test,y_test)
    print("final test accuracy: %.4f%%" % (score*100))
