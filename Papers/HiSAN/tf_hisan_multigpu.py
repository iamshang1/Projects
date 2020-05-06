import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import sys
import time
from sklearn.metrics import f1_score
import random

class hisan_multigpu(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,num_gpus=1):

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = tf.get_variable('embeddings',
                                initializer=embedding_matrix.astype(np.float32),
                                dtype=tf.float32)
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.num_gpus = num_gpus
        self.unk_tok = embedding_matrix.shape[0] - 1
        self.vocab_size = embedding_matrix.shape[0]

        with tf.variable_scope('hisan',reuse=tf.AUTO_REUSE):

            self.logits = []
            self.predictions = []

            #inputs
            self.doc_inputs = []
            self.labels = tf.placeholder(tf.int32,shape=[None])

            for g in range(self.num_gpus):
                with tf.device("/gpu:%d" % g):

                    #doc input
                    doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
                    self.doc_inputs.append(doc_input)
                    doc_embeds = tf.map_fn(self._attention_step,doc_input,dtype=tf.float32)

                    #classification functions
                    logit = tf.layers.dense(doc_embeds,num_classes,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name='logits')
                    self.logits.append(logit)
                    self.predictions.append(tf.nn.softmax(logit))

            #predictions and optimizers
            self.predictions = tf.concat(self.predictions,0)
            self.logits = tf.concat(self.logits,0)
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.logits,labels=self.labels)
            self.optimizer = tf.train.AdamOptimizer(0.0001,0.9,0.99).minimize(
                             self.loss,colocate_gradients_with_ops=True)

        #init op
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _attention_step(self,doc):

        with tf.variable_scope('hisan',reuse=tf.AUTO_REUSE):

            words_per_line = tf.count_nonzero(doc,1,dtype=tf.int32)
            num_lines = tf.count_nonzero(words_per_line,dtype=tf.int32)
            max_words_ = tf.reduce_max(words_per_line)
            doc_input_reduced = doc[:num_lines,:max_words_]
            num_words = words_per_line[:num_lines]

            #word embeddings
            word_embeds = tf.gather(self.embedding_matrix,doc_input_reduced)
            word_embeds = tf.nn.dropout(word_embeds,self.dropout)

            #masking
            mask_base = tf.cast(tf.sequence_mask(num_words,max_words_),tf.float32)
            mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
            mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,max_words_])
            mask3 = tf.tile(tf.expand_dims(mask_base,1),[self.attention_heads,1,1])

            #word self attention
            Q = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='word_Q')
            K = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='word_K')
            V = tf.layers.conv1d(word_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='word_V')

            Q = tf.multiply(Q,mask)
            K = tf.multiply(K,mask)
            V = tf.multiply(V,mask)

            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.add(outputs,(mask2-1)*1e10)
            outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
            outputs = tf.multiply(outputs,mask2)
            outputs = tf.matmul(outputs,V_)
            word_outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
            word_outputs = tf.multiply(word_outputs,mask)

            #word target attention
            Q = tf.get_variable('word_Q',(1,1,self.attention_size),
                tf.float32,tf.orthogonal_initializer())
            Q = tf.tile(Q,[num_lines,1,1])

            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(word_outputs,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(word_outputs,self.attention_heads,axis=2),axis=0)

            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.add(outputs,(mask3-1)*1e10)
            outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
            outputs = tf.matmul(outputs,V_)
            sent_embeds = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
            sent_embeds = tf.transpose(sent_embeds,[1,0,2])
            sent_embeds = tf.nn.dropout(sent_embeds,self.dropout)

            #sent self attention
            Q = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='sent_Q')
            K = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='sent_K')
            V = tf.layers.conv1d(sent_embeds,self.attention_size,1,padding='same',activation=tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),name='sent_V')

            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
            outputs = tf.matmul(outputs,V_)
            sent_outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)

            #sent target attention       
            Q = tf.get_variable('sent_Q_target',(1,1,self.attention_size),
                tf.float32,tf.orthogonal_initializer())

            Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
            K_ = tf.concat(tf.split(sent_outputs,self.attention_heads,axis=2),axis=0)
            V_ = tf.concat(tf.split(sent_outputs,self.attention_heads,axis=2),axis=0)

            outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
            outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
            outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
            outputs = tf.matmul(outputs,V_)
            doc_embed = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
            doc_embed = tf.nn.dropout(tf.squeeze(doc_embed,[0]),self.dropout)

        return tf.squeeze(doc_embed,[0])
    
    def _split_lines(self,data,noise=False):
    
        batch_size = len(data)
        retval = np.zeros((batch_size,self.ms,self.mw))
        for i,doc in enumerate(data):
            doc_ = doc
            doc_ = list(doc[doc.nonzero()])
            
            #randomly add padding to front
            if noise:
                pad_amt = np.random.randint(0,self.mw)
                doc_ = [int(self.unk_tok) for i in range(pad_amt)] + doc_
            tokens = len(doc_)

            for j,line in enumerate([doc_[i:i+self.mw] for i in range(0,tokens,self.mw)]):
                line_ = line
                l = len(line_)
                
                #randomly replace tokens
                if noise and np.count_nonzero(line) == self.mw:
                    r_idx = np.random.randint(0,self.mw)
                    line_[r_idx] = np.random.randint(1,self.vocab_size)
                retval[i,j,:l] = line_

        gpu_batch_size = int(np.ceil(batch_size/self.num_gpus))
        data_split = [retval[i:i+gpu_batch_size] for i in range(0,batch_size,gpu_batch_size)]

        return data_split
    
    def train(self,data,labels,batch_size=128,epochs=1000,patience=5,validation_data=None,
              savebest=False,filepath=None):
        
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
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            xy = list(zip(data,labels))
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = np.array(data)
            labels = np.array(labels)
            preds = []

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                data_split = self._split_lines(data[start:stop],noise=True)
                feed_dict = {self.dropout:self.dropout_keep,self.labels:labels[start:stop]}
                for g in range(self.num_gpus):
                    feed_dict[self.doc_inputs[g]] = data_split[g]
                pred,loss,_ = self.sess.run([self.predictions,self.loss,self.optimizer],
                               feed_dict=feed_dict)
                preds.append(np.argmax(pred,1))
                    
                sys.stdout.write("epoch %i, training sample %i of %i      \r"\
                                 % (ep+1,stop,len(data)))
                sys.stdout.flush()
            print()
            
            #checkpoint after every epoch
            preds = np.concatenate(preds,0)
            micro = f1_score(labels,preds,average='micro')
            macro = f1_score(labels,preds,average='macro')
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

    def predict(self,data,batch_size=128):

        preds = []
        
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            data_split = self._split_lines(data[start:stop])
            feed_dict = {self.dropout:1.0}
            for g in range(self.num_gpus):
                feed_dict[self.doc_inputs[g]] = data_split[g]
            pred = self.sess.run(self.predictions,feed_dict=feed_dict)
            preds.append(np.argmax(pred,1))
            sys.stdout.write("predicting sample %i of %i      \r" % (stop,len(data)))
            sys.stdout.flush()

        print()
        preds = np.concatenate(preds,0)
        return preds
        
    def score(self,data,labels,batch_size=128):
  
        preds = self.predict(data,batch_size=batch_size)
        micro = f1_score(labels,preds,average='micro')
        macro = f1_score(labels,preds,average='macro')
        return micro,macro

    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    '''
    dummy test data
    '''

    n_gpus = len([x.name for x in device_lib.list_local_devices() if 'GPU' in x.device_type])
    print('training on %i gpus' % n_gpus)

    #params
    batch_size = 32
    epochs = 30
    train_samples = 1000
    test_samples = 500
    vocab_size = 1000
    max_words = 200
    num_classes = 3
    embedding_size = 100
    attention_heads = 4
    attention_size = 200
    
    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X_train = []
    X_test = []
    for i in range(train_samples):
        l = np.random.randint(50,max_words)
        X = np.zeros(max_words)
        X[:l] = np.random.randint(1,vocab_size,l)
        X_train.append(X)
    for i in range(test_samples):
        l = np.random.randint(50,max_words)
        X = np.zeros(max_words)
        X[:l] = np.random.randint(1,vocab_size,l)
        X_test.append(X)
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.random.randint(0,num_classes,train_samples)
    y_test = np.random.randint(0,num_classes,test_samples)

    #make save dir
    if not os.path.exists('savedmodels'):
        os.makedirs('savedmodels')   

    #train model
    model = hisan_multigpu(vocab,num_classes,int(np.ceil(max_words/15)+1),15,
                  attention_heads,attention_size,num_gpus=n_gpus)
    model.train(X_train,y_train,batch_size,epochs,validation_data=(X_test,y_test),
                savebest=True,filepath='savedmodels/model.ckpt')
    model.load('savedmodels/model.ckpt')