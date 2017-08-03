'''
hiearchical attention transformer for document classification
https://arxiv.org/pdf/1706.03762.pdf
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
import sys
import time

class hierarchical_attention_transformer(object):
    '''
    hiearchical attention transformer for document classification
    https://arxiv.org/pdf/1706.03762.pdf
    '''
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,
                 attention_heads=8,attention_size=512,dropout_keep=0.9):

        self.attention_heads = attention_heads
        self.attention_size = attention_size
        self.vocab = embedding_matrix
        self.embedding_size = embedding_matrix.shape[1]
        self.embeddings = embedding_matrix.astype(np.float32)
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.max_words = max_words
        
        #shared variables
        self.word_atten_W = tf.Variable(self._ortho_weight(attention_size,attention_size))
        self.word_atten_b = tf.Variable(np.asarray(np.zeros(attention_size),dtype=np.float32))
        self.word_softmax = tf.Variable(self._ortho_weight(attention_size,1))
        self.sent_atten_W = tf.Variable(self._ortho_weight(attention_size,attention_size))
        self.sent_atten_b = tf.Variable(np.asarray(np.zeros(attention_size),dtype=np.float32))
        self.sent_softmax = tf.Variable(self._ortho_weight(attention_size,1))
        self.W_softmax = tf.Variable(self._ortho_weight(attention_size,num_classes))
        self.b_softmax = tf.Variable(np.asarray(np.zeros(num_classes),dtype=np.float32))
        
        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        self.sent_sum = tf.reduce_sum(self.doc_input,1)
        self.sent_mask = tf.not_equal(self.sent_sum,tf.zeros_like(self.sent_sum))
        self.sent_nonzero = tf.boolean_mask(self.doc_input,self.sent_mask)
        
        #get sent embeddings
        self.sent_embeds = tf.map_fn(self._sent_embedding_step,self.sent_nonzero,dtype=tf.float32)
        self.sent_embeds = tf.expand_dims(self.sent_embeds,0)
        self.sent_embeds = tf.nn.dropout(self.sent_embeds,self.dropout)

        #sentence attention blocks
        with tf.variable_scope("sent_block"):  
            Q = tf.layers.dense(self.sent_embeds,self.attention_size,activation=tf.nn.relu)
            K = tf.layers.dense(self.sent_embeds,self.attention_size,activation=tf.nn.relu)
            V = tf.layers.dense(self.sent_embeds,self.attention_size,activation=tf.nn.relu)
            
            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)
            
            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.nn.softmax(outputs)
            outputs = tf.nn.dropout(outputs,self.dropout)
            outputs = tf.matmul(outputs,V_)
            outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
            outputs = tf.nn.dropout(outputs,self.dropout)

        #create sentence embedding
        self.sent_embeds = tf.squeeze(outputs,[0])
        self.sent_u = tf.nn.relu(tf.matmul(self.sent_embeds,self.sent_atten_W)+self.sent_atten_b)
        self.sent_exp = tf.exp(tf.matmul(self.sent_u,self.sent_softmax))
        self.sent_atten = self.sent_exp/tf.reduce_sum(self.sent_exp)
        self.doc_embed = tf.transpose(tf.matmul(tf.transpose(self.sent_embeds),self.sent_atten))

        #classification functions
        self.output = tf.matmul(self.doc_embed,self.W_softmax)+self.b_softmax
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.99).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def _sent_embedding_step(self,sent):
        '''
        get sentence embeddings
        '''
        #get word embeddings
        word_mask = tf.not_equal(sent,tf.zeros_like(sent))
        word_nonzero = tf.boolean_mask(sent,word_mask)
        word_embeds = tf.expand_dims(tf.gather(tf.get_variable('embeddings',
                      initializer=self.embeddings,dtype=tf.float32),word_nonzero),0)
        word_embeds = tf.nn.dropout(word_embeds,self.dropout)

        #word attention blocks
        with tf.variable_scope("word_block"):
        
            #attention block
            Q = tf.layers.dense(word_embeds,self.attention_size,activation=tf.nn.relu)
            K = tf.layers.dense(word_embeds,self.attention_size,activation=tf.nn.relu)
            V = tf.layers.dense(word_embeds,self.attention_size,activation=tf.nn.relu)
            
            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)
            
            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.nn.softmax(outputs)
            outputs = tf.nn.dropout(outputs,self.dropout)
            outputs = tf.matmul(outputs,V_)
            outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
            outputs = tf.nn.dropout(outputs,self.dropout)
            
        #create sentence embedding
        word_embeds = tf.squeeze(outputs,[0])
        word_u = tf.nn.relu(tf.matmul(word_embeds,self.word_atten_W)+self.word_atten_b)
        word_exp = tf.exp(tf.matmul(word_u,self.word_softmax))
        word_atten = word_exp/tf.reduce_sum(word_exp)
        sent_embed = tf.matmul(tf.transpose(word_embeds),word_atten)
        return tf.squeeze(sent_embed)
        
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
        return W.astype(np.float32)
     
    def train(self,data,labels,epochs=30,validation_data=None,savebest=False,filepath=None):
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
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print 'training network on %i documents, validating on %i documents' \
              % (len(data), validation_size)
        
        #track best model for saving
        prevbest = 0    
        for i in range(epochs):
            correct = 0.
            start = time.time()
            
            #train
            for doc in range(data.shape[0]):
                feed_dict = {self.doc_input:data[doc],self.labels:labels[doc],self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],feed_dict=feed_dict)
                if np.argmax(pred) == np.argmax(labels[doc]):
                    correct += 1
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f      \r"\
                                 % (i+1,doc+1,data.shape[0],cost))
                sys.stdout.flush()
            print
            print "training time: %.2f" % (time.time()-start)
            trainscore = correct/len(data)
            print "epoch %i training accuracy: %.4f%%" % (i+1, trainscore*100)
            
            #validate
            if validation_data:
                score = self.score(validation_data[0],validation_data[1])
                print "epoch %i validation accuracy: %.4f%%" % (i+1, score*100)
                
            #save if performance better than previous best
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
        for doc in range(data.shape[0]):
            feed_dict = {self.doc_input:data[doc],self.dropout:1.0}
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
        for doc in range(data.shape[0]):
            feed_dict = {self.doc_input:data[doc],self.dropout:1.0}
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