import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
import sys
import time
from sklearn.metrics import f1_score
import random

class mthisan(object):

    class mthisan_model(Model):

        def __init__(self,embedding_matrix,num_classes,attention_size,attention_heads):
        
            super(mthisan.mthisan_model,self).__init__()
            self.attention_size = attention_size
            self.attention_heads = attention_heads
            
            self.embedding = layers.Embedding(embedding_matrix.shape[0],
                             embedding_matrix.shape[1],
                             embeddings_initializer=tf.keras.initializers.Constant(
                             embedding_matrix.astype(np.float32)))
            self.word_drop = layers.Dropout(0.1)
            self.word_Q = layers.Dense(self.attention_size)
            self.word_K = layers.Dense(self.attention_size)
            self.word_V = layers.Dense(self.attention_size)
            self.word_target = tf.Variable(tf.random.uniform(shape=[1,self.attention_heads,1,
                               int(self.attention_size/self.attention_heads)]))
            self.word_self_att = layers.Attention(use_scale=True)
            self.word_targ_att = layers.Attention(use_scale=True)
            
            self.line_drop = layers.Dropout(0.1)
            self.line_Q = layers.Dense(self.attention_size)
            self.line_K = layers.Dense(self.attention_size)
            self.line_V = layers.Dense(self.attention_size)
            self.line_target = tf.Variable(tf.random.uniform(shape=[1,self.attention_heads,1,
                               int(self.attention_size/self.attention_heads)]))
            self.line_self_att = layers.Attention(use_scale=True)
            self.line_targ_att = layers.Attention(use_scale=True)

            self.doc_drop = layers.Dropout(0.1)
            
            self.classify_layers = []
            for c in num_classes:
                self.classify_layers.append(layers.Dense(c))
                                 
        def call(self,docs):
            
            #input shape: batch x lines x words        
            doc_embeds = tf.map_fn(self._attention_step,docs,dtype=tf.float32)
            logits = []
            for l in self.classify_layers:
                logits.append(l(doc_embeds))
            return logits

        def _split_heads(self,x,batch_size):
            x = tf.reshape(x,(batch_size,-1,self.attention_heads,
                           int(self.attention_size/self.attention_heads)))
            return tf.transpose(x,perm=[0, 2, 1, 3])
            
        def _attention_step(self,doc):
        
            #input: lines x words
            words_per_line = tf.math.count_nonzero(doc,1)
            num_lines = tf.math.count_nonzero(words_per_line)
            max_words = tf.reduce_max(words_per_line)
            doc_input_reduced = doc[:num_lines,:max_words]
            num_words = words_per_line[:num_lines]
            mask = tf.sequence_mask(num_words,max_words)    #num_lines x max_words
            mask = tf.tile(tf.expand_dims(mask,1),[1,self.attention_heads,1])  #num_lines x heads x max_words
                        
            #word embeddings
            word_embeds = self.embedding(doc_input_reduced)  #num_lines x max_words x embed_dim
            word_embeds = self.word_drop(word_embeds)
            
            #word self attention
            word_q = self._split_heads(self.word_Q(word_embeds),num_lines)   #num_lines x heads x max_words x depth
            word_k = self._split_heads(self.word_K(word_embeds),num_lines)   #num_lines x heads x max_words x depth
            word_v = self._split_heads(self.word_V(word_embeds),num_lines)   #num_lines x heads x max_words x depth
            word_self_out = self.word_self_att([word_q,word_v,word_k],[mask,mask])
            
            #word target attention
            word_target = tf.tile(self.word_target,[num_lines,1,1,1])       #num_lines x heads x 1 x depth
            word_targ_out = self.word_targ_att([word_target,word_self_out,word_self_out],[None,mask])
            word_targ_out = tf.transpose(word_targ_out,perm=[0, 2, 1, 3])   #num_lines x 1 x heads x depth
            line_embeds = tf.reshape(word_targ_out,(num_lines,1,self.attention_size))
            line_embeds = tf.expand_dims(tf.squeeze(line_embeds,[1]),0)     #1 x num_lines x attention_size
            line_embeds = self.line_drop(line_embeds)
            
            #line self attention
            line_q = self._split_heads(self.line_Q(line_embeds),1)   #1 x heads x num_lines x depth
            line_k = self._split_heads(self.line_K(line_embeds),1)   #1 x heads x num_lines x depth
            line_v = self._split_heads(self.line_V(line_embeds),1)   #1 x heads x num_lines x depth
            line_self_out = self.line_self_att([line_q,line_v,line_k])
            
            #word target attention
            line_targ_out = self.line_targ_att([self.line_target,line_self_out,line_self_out])
            line_targ_out = tf.transpose(line_targ_out,perm=[0, 2, 1, 3])   #1 x 1 x heads x depth
            doc_embed = tf.reshape(line_targ_out,(1,self.attention_size))
            doc_embed = self.doc_drop(doc_embed)
            
            return tf.squeeze(doc_embed,[0])
            
    def __init__(self,embedding_matrix,num_classes,max_sents=201,max_words=15,
                 attention_heads=8,attention_size=400):
    
        self.ms = max_sents
        self.mw = max_words
        self.unk_tok = embedding_matrix.shape[0] - 1
        self.num_classes = num_classes
        self.num_tasks = len(num_classes)
        self.vocab_size = embedding_matrix.shape[0]
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            self.model = self.mthisan_model(embedding_matrix,num_classes,attention_size,attention_heads)
            self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                               reduction=tf.keras.losses.Reduction.NONE)
            self.optimizer = tf.keras.optimizers.Adam(0.0001,0.9,0.99,1e-07,True)
        
        print ('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    @tf.function
    def _reshape_input(self,text,noise):
        if noise:
            l1 = tf.random.uniform([],0,self.mw,tf.int32)
            l2 = self.ms*self.mw - text.shape[1] - l1
            text = tf.pad(text,[[0,0],[l1,0]],constant_values=self.unk_tok)
            text = tf.pad(text,[[0,0],[0,l2]])
        else:
            l = self.ms*self.mw - text.shape[1]
            text = tf.pad(text,tf.constant([[0,0],[0,l]]))
        return tf.reshape(text,(-1,self.ms,self.mw))
        
    @tf.function
    def _distributed_reshape_input(self,text,noise=False):
        return self.strategy.experimental_run_v2(self._reshape_input,args=(text,noise))
    
    @tf.function
    def _distributed_transpose(self,y):
        return self.strategy.experimental_run_v2(lambda y:tf.transpose(y),args=(y,))
    
    @tf.function
    def _train_step(self,inputs,batch_size):
        text,labels = inputs
        with tf.GradientTape() as tape:
            predictions = self.model(text,training=True)
            losses = 0
            for i in range(self.num_tasks):
                loss = self.loss_object(labels[i],predictions[i])
                loss = tf.nn.compute_average_loss(loss,global_batch_size=batch_size)
                losses += loss/self.num_tasks
        gradients = tape.gradient(losses,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        return predictions,losses
        
    @tf.function
    def _distributed_train_step(self,dataset,batch_size):
        predictions,loss = self.strategy.experimental_run_v2(self._train_step,args=(dataset,batch_size))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,loss,axis=None)
        return predictions,loss
        
    @tf.function
    def _score_step(self,inputs,batch_size):
        text,labels = inputs
        predictions = self.model(text,training=False)
        losses = 0
        for i in range(self.num_tasks):
            loss = self.loss_object(labels[i],predictions[i])
            loss = tf.nn.compute_average_loss(loss,global_batch_size=batch_size)
            losses += loss/self.num_tasks
        return predictions,losses
        
    @tf.function
    def _distributed_score_step(self,dataset,batch_size):
        predictions,loss = self.strategy.experimental_run_v2(self._score_step,args=(dataset,batch_size))
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,loss,axis=None)
        return predictions,loss
        
    @tf.function
    def _predict_step(self,text):
        predictions = self.model(text,training=False)
        return predictions
        
    @tf.function
    def _distributed_predict_step(self,dataset):
        return self.strategy.experimental_run_v2(self._predict_step,args=(dataset,))

    def train(self,data,labels,batch_size=128,epochs=100,patience=5,
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

            train_dataset = tf.data.Dataset.from_tensor_slices(
                            (data,np.array(labels).T)).shuffle(len(data)).batch(batch_size) 
            train_dist_dataset = self.strategy.experimental_distribute_dataset(train_dataset)

            #shuffle data
            y_preds = [[] for c in self.num_classes]
            y_trues = [[] for c in self.num_classes]
            start_time = time.time()

            #train
            for i,batch in enumerate(train_dist_dataset):

                #train step
                X,y = batch
                y = self._distributed_transpose(y)
                X = self._distributed_reshape_input(X,noise=True)
                predictions,loss = self._distributed_train_step((X,y),batch_size)
                predictions = [tf.concat(task.values,0) for task in predictions]

                #track correct predictions
                for t,(p,l) in enumerate(zip(predictions,tf.concat(y.values,1))):
                    y_preds[t].extend(np.argmax(p,1))
                    y_trues[t].extend(l)
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,(i+1)*batch_size,len(data),loss))
                sys.stdout.flush()
                
            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            for i in range(self.num_tasks):
                micro = f1_score(y_trues[i],y_preds[i],average='micro')
                macro = f1_score(y_trues[i],y_preds[i],average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1,i,micro,macro))

            scores,loss = self.score(validation_data[0],validation_data[1],
                                     batch_size=batch_size)

            for i in range(self.num_tasks):
                print("epoch %i validation micro/macro: %.4f, %.4f" % (ep+1,scores[i][0],scores[i][1]))

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

    def predict(self,data,batch_size=128):
    
        dataset = tf.data.Dataset.from_tensor_slices((data)).batch(batch_size) 
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)    
        y_preds = [[] for c in self.num_classes]
        
        for i,X in enumerate(dist_dataset):

            #predict step
            X = self._distributed_reshape_input(X)
            predictions = self._distributed_predict_step(X)
            predictions = [tf.concat(task.values,0) for task in predictions]
            for t,p in enumerate(self.predictions):
                y_preds[t].extend(np.argmax(p,1))
            
            sys.stdout.write("processed %i of %i records        \r" \
                             % ((i+1)*batch_size,len(data)))
            sys.stdout.flush()
        print()
        return y_preds

    def score(self,data,labels,batch_size=128):
    
        dataset = tf.data.Dataset.from_tensor_slices((data,np.array(labels).T)).batch(batch_size) 
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)    
        y_preds = [[] for c in self.num_classes]
        losses = []
        
        for i,batch in enumerate(dist_dataset):

            #score step
            X,y = batch
            y = self._distributed_transpose(y)
            X = self._distributed_reshape_input(X)
            predictions,loss = self._distributed_score_step((X,y),batch_size)
            predictions = [tf.concat(task.values,0) for task in predictions]

            for t,p in enumerate(predictions):
                y_preds[t].extend(np.argmax(p,1))
            losses.append(loss)
            sys.stdout.write("processed %i of %i records        \r" \
                             % ((i+1)*batch_size,len(data)))
            sys.stdout.flush()
        
        scores = []
        for i in range(self.num_tasks):
            micro = f1_score(labels[i],y_preds[i],average='micro')
            macro = f1_score(labels[i],y_preds[i],average='macro')
            scores.append([micro,macro])
        
        return scores,np.mean(losses)
        
    def save(self,savedir):

        self.model.save(savedir)

    def load(self,savedir):

        self.model = tf.keras.models.load_model(savedir)


if __name__ == "__main__":

    '''
    dummy test data
    '''

    #params
    batch_size = 32
    epochs = 5
    train_samples = 2000
    test_samples = 2000
    vocab_size = 750
    max_words = 500
    num_classes = [2,5,10]
    embedding_size = 100
    attention_heads = 4
    attention_size = 64
    
    #create data
    vocab = np.random.rand(vocab_size,embedding_size)
    X = np.random.randint(1,vocab_size,(train_samples+test_samples,max_words))

    #test train split
    X_train = X[:train_samples]
    X_test = X[train_samples:]
    y_trains = []
    for i in num_classes:
        y_trains.append(np.random.randint(0,i,train_samples))
    y_tests = []
    for i in num_classes:
        y_tests.append(np.random.randint(0,i,test_samples))

    #make save dir
    if not os.path.exists('mymodel'):
        os.makedirs('mymodel')   

    #train model
    model = mthisan(vocab,num_classes,int(np.ceil(max_words/15)+1),15,
                    attention_heads,attention_size)
    model.train(X_train,y_trains,batch_size,epochs,validation_data=(X_test,y_tests),
                savebest=True,filepath='mymodel')