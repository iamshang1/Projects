import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics import f1_score
import random

class hisan(object):

    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,activation=tf.nn.elu,lr=0.0001):

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.activation = activation
        self.num_tasks = len(num_classes)

        #doc input
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
        self.doc_embeds = tf.map_fn(self._attention_step,self.doc_input,dtype=tf.float32)

        #classification functions
        self.logits = []
        self.predictions = []
        for t in range(self.num_tasks):
            logit = tf.layers.dense(self.doc_embeds,num_classes[t],
                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.logits.append(logit)
            self.predictions.append(tf.nn.softmax(logit))

        #loss, accuracy, and training functions
        self.labels = []
        self.loss = 0
        for t in range(self.num_tasks):
            label = tf.placeholder(tf.int32,shape=[None])
            self.labels.append(label)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                  logits=self.logits[t],labels=label))
            self.loss += loss/self.num_tasks                                      
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _attention_step(self,doc):

        words_per_line = tf.math.count_nonzero(doc,1)
        num_lines = tf.math.count_nonzero(words_per_line)
        max_words_ = tf.reduce_max(words_per_line)
        doc_input_reduced = doc[:num_lines,:max_words_]
        num_words = words_per_line[:num_lines]

        #word embeddings
        word_embeds = tf.gather(tf.get_variable('embeddings',
                      initializer=self.embedding_matrix,dtype=tf.float32),
                      doc_input_reduced)
        word_embeds = tf.nn.dropout(word_embeds,self.dropout)

        #masking
        mask_base = tf.cast(tf.sequence_mask(num_words,max_words_),tf.float32)
        mask = tf.tile(tf.expand_dims(mask_base,2),[1,1,self.attention_size])
        mask2 = tf.tile(tf.expand_dims(mask_base,2),[self.attention_heads,1,max_words_])

        #word self attention
        Q = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.where(tf.equal(mask,0),tf.zeros_like(Q),Q)
        K = tf.where(tf.equal(mask,0),tf.zeros_like(K),K)
        V = tf.where(tf.equal(mask,0),tf.zeros_like(V),V)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        word_self = tf.where(tf.equal(mask2,0),tf.zeros_like(outputs),outputs)
        outputs = tf.matmul(word_self,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        outputs = tf.where(tf.equal(mask,0),tf.zeros_like(outputs),outputs)

        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        Q = tf.tile(Q,[num_lines,1,1])

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.where(tf.equal(outputs,0),tf.ones_like(outputs)*-1000,outputs)
        word_target = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(word_target,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        sent_embeds = tf.transpose(outputs,[1,0,2])
        sent_embeds = tf.nn.dropout(sent_embeds,self.dropout)

        #sent self attention
        Q = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        sent_self = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(sent_self,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)

        #sent target attention
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        sent_target = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.matmul(sent_target,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        doc_embed = tf.nn.dropout(tf.squeeze(outputs,[0]),self.dropout)
        doc_embed = tf.squeeze(doc_embed,[0])

        return doc_embed
    
    def train(self,data,labels,batch_size=64,epochs=30,patience=5,
              validation_data=None,savebest=False,filepath=None):
        
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
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            labels.append(data)
            xy = list(zip(*labels))
            random.shuffle(xy)
            shuffled = list(zip(*xy))
            data = list(shuffled[-1])
            labels = list(shuffled[:self.num_tasks])

            y_preds = [[] for i in range(self.num_tasks)]
            y_trues = [[] for i in range(self.num_tasks)]
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                feed_dict = {self.doc_input:data[start:stop],self.dropout:self.dropout_keep}
                for t in range(self.num_tasks):
                    feed_dict[self.labels[t]] = labels[t][start:stop]
                retvals = self.sess.run(self.predictions + [self.optimizer,self.loss],
                                        feed_dict=feed_dict)
                loss = retvals[-1]

                #track correct predictions
                for t in range(self.num_tasks):
                    y_preds[t].extend(np.argmax(retvals[t],1))
                    y_trues[t].extend(labels[t][start:stop])
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop,len(data),loss))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            for t in range(self.num_tasks):
                micro = f1_score(y_trues[t],y_preds[t],average='micro')
                macro = f1_score(y_trues[t],y_preds[t],average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1,t+1,micro,macro))

            scores,loss = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            for t in range(self.num_tasks):
                micro,macro = scores[t]
                print("epoch %i task %i validation micro/macro: %.4f, %.4f" % (ep+1,t+1,micro,macro))
            print("epoch %i validation loss: %.8f" % (ep+1,loss))

            #save if performance better than previous best
            if loss < bestloss:
                bestloss = loss
                pat_count = 0
                if savebest:
                    self.save(filepath)
            else:
                pat_count += 1
                if pat_count >= patience:
                    break

            #reset timer
            start_time = time.time()

    def score(self,data,labels,batch_size=64):
        
        y_preds = [[] for t in range(self.num_tasks)]
        loss = []
        for start in range(0,len(data),batch_size):
        
            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            feed_dict = {self.doc_input:data[start:stop],self.dropout:1.0}
            for t in range(self.num_tasks):
                feed_dict[self.labels[t]] = labels[t][start:stop]
            preds = self.sess.run(self.predictions + [self.loss],feed_dict=feed_dict)
            loss.append(preds[-1])
            for t in range(self.num_tasks):
                y_preds[t].append(np.argmax(preds[t],1))
                
            sys.stdout.write("processed %i of %i records        \r" % (stop,len(data)))
            sys.stdout.flush()
            
        print()
        for t in range(self.num_tasks):
            y_preds[t] = np.concatenate(y_preds[t],0)
            
        scores = []
        for t in range(self.num_tasks):  
            micro = f1_score(labels[t],y_preds[t],average='micro')
            macro = f1_score(labels[t],y_preds[t],average='macro')
            scores.append((micro,macro))
        
        return scores,np.mean(loss)

    def save(self,filename):

        self.saver.save(self.sess,filename)

    def load(self,filename):

        self.saver.restore(self.sess,filename)

if __name__ == "__main__":

    #params
    batch_size = 32
    epochs = 5
    train_samples = 2000
    test_samples = 2000
    vocab_size = 750
    max_lines = 50
    max_words = 10
    num_classes = [2,5,10]
    embedding_size = 100
    attention_heads = 4
    attention_size = 64
    
    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(1,vocab_size,
        (train_samples+test_samples,max_lines,max_words))

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = []
    for i in num_classes:
        y_trains.append(np.random.randint(0,i,train_samples))
    y_tests = []
    for i in num_classes:
        y_tests.append(np.random.randint(0,i,test_samples))

    #train model
    model = hisan(vocab,num_classes,max_lines,max_words,
                  attention_heads,attention_size)
    model.train(X_train,y_trains,batch_size,epochs,
                validation_data=(X_test,y_tests))
