import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from tensorflow.contrib.layers import xavier_initializer 
import sys
import threading
import time
import random
import copy

class hierarchical_attention_network(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,rnn_type="gru",
                 rnn_units=200,attention_context=300,dropout_keep=0.5,gpus=1):

        self.rnn_units = rnn_units
        self.attention_context = attention_context
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.dropout_keep = dropout_keep
        self.vocab = embedding_matrix
        self.embeddings = tf.constant(self.vocab,tf.float32)
        self.embedding_size = embedding_matrix.shape[1]
        self.gpus = gpus
        self.max_sents = max_sents
        self.max_words = max_words

        #central variables and ops on cpu
        with tf.device("/cpu:0"):
        
            with tf.variable_scope('master'):
                
                self.dropout = tf.placeholder(tf.float32)
        
                #doc input and line count
                self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
                num_lines = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(self.doc_input,2),0),tf.int32),1)
                max_lines = tf.reduce_max(num_lines)
                
                #get sent embeddings
                doc_input_timemajor = tf.transpose(self.doc_input,[1,0,2])[:max_lines]
                sent_embeds = tf.map_fn(self._sent_embedding_step,doc_input_timemajor,dtype=tf.float32)
                sent_embeds = tf.transpose(sent_embeds,[1,0,2])
            
                #sentence rnn layer
                with tf.variable_scope('sentence'):
                    [sent_outputs_fw,sent_outputs_bw],_ = tf.nn.bidirectional_dynamic_rnn(
                                                          self.rnn_cell(rnn_units),self.rnn_cell(rnn_units),
                                                          sent_embeds,sequence_length=num_lines,dtype=tf.float32)
                sent_outputs = tf.concat((sent_outputs_fw,sent_outputs_bw),2)
            
                #sentence attention
                u = tf.nn.tanh(tf.matmul(tf.reshape(sent_outputs,[-1,rnn_units*2]),
                    tf.get_variable('sent_W',(2*rnn_units,attention_context),tf.float32,xavier_initializer())))
                exps = tf.exp(tf.matmul(u,
                       tf.get_variable('sent_context',(attention_context,1),tf.float32,xavier_initializer())))
                self.alpha = tf.reshape(exps,[-1,max_lines,1])
                self.alpha /= tf.reshape(tf.reduce_sum(self.alpha,1)+0.000000001,[-1,1,1])
                self.doc_embed = tf.reduce_sum(sent_outputs*self.alpha,1)
                doc_embed_drop = tf.nn.dropout(self.doc_embed,self.dropout)
                
                #classification
                self.prediction = tf.nn.softmax(tf.matmul(doc_embed_drop,
                                  tf.get_variable('softmax_W',(rnn_units*2,num_classes),tf.float32,xavier_initializer()))+\
                                  tf.get_variable('softmax_b',(num_classes),tf.float32,tf.zeros_initializer()))
        
        #store local placeholders and return values in lists
        self.doc_inputs = []
        self.labels = []
        self.predictions = []
        self.loss = []
        self.optimizer = []
        self.elastic_ops = []

        #local variables and ops on gpus
        for i in range(gpus):
            with tf.device("/gpu:%d" % i):

                with tf.variable_scope('local_%i' % i):

                    #doc input and line count
                    self.doc_inputs.append(tf.placeholder(tf.int32, shape=[None,max_sents,max_words]))
                    num_lines = tf.reduce_sum(tf.cast(tf.greater(tf.reduce_sum(self.doc_inputs[i],2),0),tf.int32),1)
                    max_lines = tf.reduce_max(num_lines)
                    
                    #get sent embeddings
                    doc_input_timemajor = tf.transpose(self.doc_inputs[i],[1,0,2])[:max_lines]
                    sent_embeds = tf.map_fn(self._sent_embedding_step,doc_input_timemajor,dtype=tf.float32)
                    sent_embeds = tf.transpose(sent_embeds,[1,0,2])
                
                    #sentence rnn layer
                    with tf.variable_scope('sentence'):
                        [sent_outputs_fw,sent_outputs_bw],_ = tf.nn.bidirectional_dynamic_rnn(
                                                              self.rnn_cell(rnn_units),self.rnn_cell(rnn_units),
                                                              sent_embeds,sequence_length=num_lines,dtype=tf.float32)
                    sent_outputs = tf.concat((sent_outputs_fw,sent_outputs_bw),2)
                
                    #sentence attention
                    u = tf.nn.tanh(tf.matmul(tf.reshape(sent_outputs,[-1,rnn_units*2]),
                        tf.get_variable('sent_W',(2*rnn_units,attention_context),tf.float32,xavier_initializer())))
                    exps = tf.exp(tf.matmul(u,
                           tf.get_variable('sent_context',(attention_context,1),tf.float32,xavier_initializer())))
                    alpha = tf.reshape(exps,[-1,max_lines,1])
                    alpha /= tf.reshape(tf.reduce_sum(alpha,1)+0.000000001,[-1,1,1])
                    doc_embed = tf.reduce_sum(sent_outputs*alpha,1)
                    doc_embed_drop = tf.nn.dropout(doc_embed,self.dropout)
                    
                    #loss, accuracy, and training functions
                    output = tf.matmul(doc_embed_drop,
                             tf.get_variable('softmax_W',(rnn_units*2,num_classes),tf.float32,xavier_initializer()))+\
                             tf.get_variable('softmax_b',(num_classes),tf.float32,tf.zeros_initializer())
                    self.predictions.append(tf.nn.softmax(output))
                    self.labels.append(tf.placeholder(tf.float32, shape=[None,num_classes]))
                    self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=self.labels[i])))
                    self.optimizer.append(tf.train.AdamOptimizer().minimize(self.loss[i]))
        
                #elastic averaging operations
                self.elastic_ops.append([])
                master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"master")
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"local_%i" % i)
                for master_var,local_var in zip(master_vars,local_vars):
                    diff = local_var - master_var
                    self.elastic_ops[i].append(master_var.assign_add(1./gpus * diff))
                    self.elastic_ops[i].append(local_var.assign_add(-1./gpus * diff))
            
        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        
        #config options
        self.config = tf.ConfigProto()
        #self.config.log_device_placement = True
        self.config.allow_soft_placement = True
        
        #run session
        self.sess = tf.Session(config=self.config)
        self.sess.run(self.init_op)

        #sync all local variables to master
        for i in range(gpus):
            master_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"master")
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"local_%i" % i)
            for master_var,local_var in zip(master_vars,local_vars):
                self.sess.run(local_var.assign(master_var))

    def _sent_embedding_step(self,line):
        
        #embedding lookup
        num_words = tf.reduce_sum(tf.cast(tf.greater(line,0),tf.int32),1)
        max_words = tf.reduce_max(num_words)
        line = line[:,:max_words]
        word_embeds = tf.gather(tf.get_variable('embeddings',initializer=self.embeddings),line)
        
        #word rnn layer
        with tf.variable_scope('words'):
            [word_outputs_fw,word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    word_embeds,sequence_length=num_words,dtype=tf.float32)
        word_outputs = tf.concat((word_outputs_fw, word_outputs_bw),2)
        
        #word attention
        u = tf.nn.tanh(tf.matmul(tf.reshape(word_outputs,[-1,self.rnn_units*2]),
            tf.get_variable('word_W',(2*self.rnn_units,self.attention_context),tf.float32,xavier_initializer())))
        exps = tf.exp(tf.matmul(u,
               tf.get_variable('word_context',(self.attention_context,1),tf.float32,xavier_initializer())))
        alpha = tf.reshape(exps,[-1,max_words,1])
        alpha /= tf.reshape((tf.reduce_sum(alpha,1)+0.000000001),[-1,1,1])
        sent_embed = tf.reduce_sum(word_outputs*alpha,1)
        return sent_embed
        
    def _train_op(self,gpu,epoch,datalen,batch_size):
        while len(self.tempdata) > 0:
        
            #pop batch size from data and labels
            data = self._list_to_numpy(self.tempdata[-batch_size:],batch_size)
            self.tempdata = self.tempdata[:-batch_size]
            label = self.templabels[-batch_size:]
            self.templabels = self.templabels[:-batch_size]
            
            #in case batch size exceeds list size 
            if len(data) == 0:
                data =  self._list_to_numpy(self.tempdata[:],batch_size)
                self.tempdata = self.tempdata[:0]
                label = self.templabels[:]
                self.templabels = self.templabels[:0]
            
            feed_dict = {self.doc_inputs[gpu]:data,self.labels[gpu]:label,self.dropout:self.dropout_keep}
            pred,cost,_ = self.sess.run([self.predictions[gpu],self.loss[gpu],self.optimizer[gpu]],feed_dict=feed_dict)
            self.sess.run(self.elastic_ops[gpu])
            
            #track corect and processed
            self.correct += np.sum(np.argmax(pred,1)==np.argmax(label,1))
            self.processed += len(label)
            sys.stdout.write("epoch %i, processed %i of %i, loss: %f      \r"\
                             % (epoch+1,self.processed,datalen,cost))
            sys.stdout.flush()
    
    def _list_to_numpy(self,inputval,batch_size):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros((batch_size,self.max_sents,self.max_words))
            for i,doc in enumerate(inputval):
                for j,line in enumerate(doc):
                    for k, word in enumerate(line):
                        retval[i,j,k] = word
            return retval
        elif type(inputval) == np.ndarray:
            return inputval
        else:
            raise Exception("invalid input type")
    
    def train(self,data,labels,batch_size=30,epochs=30,validation_data=None,savebest=False,
              filepath=None):
  
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")
        
        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)
        
        print 'training network on %i documents, validating on %i documents' \
              % (len(data), validation_size)
        
        prevbest = 0    
        for epoch in range(epochs):
            self.correct = 0.
            self.processed = 0
            start = time.time()
            
            #shuffle
            combined = list(zip(data, labels))
            random.shuffle(combined)
            data[:], labels[:] = zip(*combined)
            
            #save data into temporary shared variable
            self.tempdata = data[:]
            self.templabels = labels[:]
            
            #run on separate gpu threads
            train_threads = []
            for gpu in range(self.gpus):
                train_threads.append(threading.Thread(target=self._train_op,
                                     args=(gpu,epoch,len(data),batch_size)))
            for t in train_threads:
              t.start()
              time.sleep(0.5)
            for t in train_threads:
              t.join()
              time.sleep(0.5)

            print ""
            print "time taken: ", time.time() - start
            trainscore = self.correct/len(data)
            print "epoch %i training accuracy: %.4f%%" % (epoch+1, trainscore*100)

            if validation_data:
                score = self.score(validation_data[0],validation_data[1],batch_size)
                print "epoch %i validation accuracy: %.4f%%" % (epoch+1, score*100)
            if savebest and score >= prevbest:
                prevbest = score
                self.save(filepath)
        
    def score(self,data,labels,batch_size=30):
    
        correct = 0.
        
        for i in range(len(data)):
            doc = self._list_to_numpy(data[i:i+1],1)
            label = labels[i:i+1]
        
            #get prediction
            feed_dict = {self.doc_input:doc,self.dropout:1.0}
            pred = self.sess.run(self.prediction,feed_dict=feed_dict)
            
            #track corect and processed
            correct += np.sum(np.argmax(pred,1)==np.argmax(label,1))
        
        accuracy = correct/len(labels)
        
        return accuracy
        
    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    from sklearn.preprocessing import LabelEncoder, LabelBinarizer
    from sklearn.model_selection import train_test_split
    import pickle
    import os

    #load saved files
    print "loading data"
    vocab = np.load('embeddings.npy')
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    num_docs = len(data)

    #convert data to numpy arrays
    print "converting data to arrays"
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
    print "training hierarchical attention network"
    nn = hierarchical_attention_network(vocab,classes,max_sents,max_words,gpus=2)
    nn.train(X_train,y_train,epochs=2,validation_data=(X_test,y_test),batch_size=30)

    test_acc = nn.score(X_test,y_test)
    print 'final test accuracy: %.4f%%' % (test_acc*100)
