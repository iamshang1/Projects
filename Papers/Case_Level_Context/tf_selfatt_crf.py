import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood, crf_decode
import sys
import time
from sklearn.metrics import f1_score
import random

class crf(object):

    def __init__(self,num_classes,max_docs,input_size,attention_size=300,attention_heads=6,
                 dropout_keep=0.9,lr=0.0001,enable_mask=False,pos_embeds=False):
        
        self.max_docs = max_docs
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)

        self.doc_input = tf.placeholder(tf.float32, shape=[None,max_docs,input_size])
        self.num_docs = tf.placeholder(tf.int32, shape=[None])
        max_len = tf.reduce_max(self.num_docs)
        doc_input_reduced = self.doc_input[:,:max_len,:]
        doc_input_reduced = tf.nn.dropout(doc_input_reduced,self.dropout)
        batch_size = tf.reduce_sum(tf.ones_like(self.num_docs))

        self.labels = tf.placeholder(tf.int32,shape=[None,max_docs])
        labels_reduced = self.labels[:,:max_len]

        if pos_embeds:
            positions = tf.tile(tf.expand_dims(tf.range(max_len),0),[batch_size,1])
            pos_embeds = tf.gather(tf.get_variable('pos_embeds',shape=(self.max_docs,input_size),
                         dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.1)),positions)
            doc_input_reduced += pos_embeds

        doc_input_reduced = tf.nn.dropout(doc_input_reduced,self.dropout)

        #masking
        mask_base = tf.cast(tf.sequence_mask(self.num_docs,max_len),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,max_len])

        #self attention
        Q = tf.layers.conv1d(doc_input_reduced,self.attention_size,1,padding='same',
            activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(doc_input_reduced,self.attention_size,1,padding='same',
            activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(doc_input_reduced,self.attention_size,1,padding='same',
            activation=tf.nn.elu,kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.where(tf.equal(mask,0),tf.zeros_like(Q),Q)
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)
        V = tf.where(tf.equal(mask,0),tf.zeros_like(V),V)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)

        #forward mask
        if enable_mask == True:
            f_mask = tf.cast(tf.sequence_mask(tf.range(max_len)+1,max_len),tf.float32)
            f_mask = tf.tile(tf.expand_dims(f_mask,0),[batch_size*self.attention_heads,1,1])
            outputs = tf.where(tf.equal(f_mask,0),tf.zeros_like(outputs),outputs)

        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs),outputs)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        outputs = tf.where(tf.equal(mask,0),tf.zeros_like(outputs),outputs)
        outputs = tf.nn.dropout(outputs,self.dropout)
        
        #conditional random field
        weights = tf.get_variable("weights",[outputs.shape[2],num_classes],
                  initializer=tf.contrib.layers.xavier_initializer())
        matricized_docs = tf.reshape(outputs,[-1,outputs.shape[2]])
        matricized_unary = tf.matmul(matricized_docs,weights)
        self.unary_scores = tf.reshape(matricized_unary,[-1,max_len,num_classes])
        
        log_likelihood, self.transition_params = crf_log_likelihood(
                        self.unary_scores,labels_reduced,self.num_docs)
        preds,viterbi_score = crf_decode(self.unary_scores,self.transition_params,self.num_docs)

        self.doc_idx = tf.placeholder(tf.int32, shape=[None,2])
        self.prediction = tf.gather_nd(preds,self.doc_idx)
        self.unary_preds = tf.gather_nd(tf.nn.softmax(self.unary_scores),self.doc_idx)

        #loss, accuracy, and training functions
        self.loss = tf.reduce_mean(-log_likelihood)
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _batch_prepro(self,data,labels=None):
        
        batch_size = len(data)
        dims = len(data[0][0])
        retval = np.zeros((batch_size,self.max_docs,dims))
        doc_idx = []
        num_docs = []
        for i,case in enumerate(data):  
            l = len(case)
            for j in range(l):
                doc_idx.append([i,j]) 
            retval[i,:l,:] = np.array(case)
            num_docs.append(l)
        doc_idx = np.array(doc_idx)        

        if type(labels) != type(None):
            labels_mat = np.zeros((batch_size,self.max_docs))
            for i,group in enumerate(labels):
                l = len(group)
                labels_mat[i,:l] = group
            labels_flat = [label for group in labels for label in group]
            return retval,labels_mat,labels_flat,num_docs,doc_idx

        return retval,num_docs,doc_idx
    
    def train(self,data,labels,batch_size=100,epochs=50,patience=5,validation_data=None,
              savebest=False,filepath=None,val_micro=False):
        
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        #track best model for saving
        bestloss = np.inf
        prevbest = 0
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            xy = list(zip(data,labels))            
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = list(data)
            labels = list(labels)

            y_preds = []
            y_trues = []
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                X,y,y_flat,num_docs,doc_idx = self._batch_prepro(data[start:stop],labels[start:stop])
                feed_dict = {self.doc_input:X,
                             self.labels:y,
                             self.num_docs:num_docs,
                             self.doc_idx:doc_idx,
                             self.dropout:self.dropout_keep}
                pred,cost,_ = self.sess.run([self.prediction,self.loss,self.optimizer],
                                            feed_dict=feed_dict)

                #track correct predictions
                y_preds.extend(pred)
                y_trues.extend(y_flat)
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop+1,len(data),cost))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            micro = f1_score(y_trues,y_preds,average='micro')
            macro = f1_score(y_trues,y_preds,average='macro')
            print("epoch %i training micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            micro,macro,loss = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            print("epoch %i validation micro/macro: %.4f, %.4f" % (ep+1,micro,macro))

            #reset timer
            start_time = time.time()

            #save if performance better than previous best
            if val_micro and micro >= prevbest:
                prevbest = micro
                pat_count = 0
                if savebest:
                    self.save(filepath)
            elif not val_micro and loss < bestloss:
                bestloss = loss
                pat_count = 0
                if savebest:
                    self.save(filepath)
            else:
                pat_count += 1
                if pat_count >= patience:
                    break

    def predict(self,data,batch_size=100):
    
        y_preds = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            X,num_docs,doc_idx = self._batch_prepro(data[start:stop])
            feed_dict = {self.doc_input:X,self.num_docs:num_docs,self.doc_idx:doc_idx,self.dropout:1.0}
            preds = self.sess.run(self.prediction,feed_dict=feed_dict)
            y_preds.extend(preds)

            sys.stdout.write("processed %i of %i records        \r" % (stop+1,len(data)))
            sys.stdout.flush()

        print()
        return y_preds

    def score(self,data,labels,batch_size=100):
        
        y_preds = []
        y_trues = []
        losses = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            X,y,y_flat,num_docs,doc_idx = self._batch_prepro(data[start:stop],labels[start:stop])
            feed_dict = {self.doc_input:X,self.labels:y,self.num_docs:num_docs,
                         self.doc_idx:doc_idx,self.dropout:1.0}
            preds,loss = self.sess.run([self.prediction,self.loss],feed_dict=feed_dict)
            y_preds.extend(preds)
            y_trues.extend(y_flat)
            losses.append(loss)

            sys.stdout.write("processed %i of %i records        \r" % (stop+1,len(data)))
            sys.stdout.flush()

        print()
        micro = f1_score(y_trues,y_preds,average='micro')
        macro = f1_score(y_trues,y_preds,average='macro')
        loss = np.mean(losses)

        return micro,macro,loss

    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    #params
    batch_size = 64
    doc_embed_dim = 100
    max_seq_len = 10
    num_sequences = 5000
    num_classes = 5
    epochs = 10
    
    #generate dummy data
    X = []
    y = []
    for i in range(num_sequences):
        seq_len = np.random.randint(2,max_seq_len)
        X_seq = np.random.rand(seq_len,doc_embed_dim)
        y_seq = np.random.randint(0,num_classes,seq_len)
        X.append(X_seq)
        y.append(y_seq)
        
    #dummy train test split
    val_size = int(0.2 * num_sequences)
    X_train = X[:-val_size]
    X_val = X[-val_size:]
    y_train = y[:-val_size]
    y_val = y[-val_size:]
    
    #test model
    model = rnncrf(num_classes,max_seq_len,doc_embed_dim)
    model.train(X_train,y_train,batch_size,epochs,validation_data=(X_val,y_val))
