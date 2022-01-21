import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from dense_attention import scaled_attention
import sys
import time
from sklearn.metrics import f1_score
import random
#tf.enable_eager_execution()

class hisan(object):

    class hisan_model(Model):

        def __init__(self,embedding_matrix,num_classes,attention_size,attention_heads):
        
            super(hisan.hisan_model,self).__init__()
            self.attention_size = attention_size
            self.attention_heads = attention_heads
            self.training = True
            
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
            self.word_self_att = scaled_attention(use_scale=1/np.sqrt(attention_size),dropout=0.1)
            self.word_targ_att = scaled_attention(use_scale=1/np.sqrt(attention_size),dropout=0.1)
            
            self.line_drop = layers.Dropout(0.1)
            self.line_Q = layers.Dense(self.attention_size)
            self.line_K = layers.Dense(self.attention_size)
            self.line_V = layers.Dense(self.attention_size)
            self.line_target = tf.Variable(tf.random.uniform(shape=[1,self.attention_heads,1,
                               int(self.attention_size/self.attention_heads)]))
            self.line_self_att = scaled_attention(use_scale=1/np.sqrt(attention_size),dropout=0.1)
            self.line_targ_att = scaled_attention(use_scale=1/np.sqrt(attention_size),dropout=0.1)

            self.doc_drop = layers.Dropout(0.1)
            self.classify = layers.Dense(num_classes)
                                 
        def call(self,docs):
            
            #input shape: batch x lines x words
            batch_size = tf.shape(docs)[0]
            words_per_line = tf.math.count_nonzero(docs,2,dtype=tf.int32)
            max_words = tf.reduce_max(words_per_line)
            lines_per_doc = tf.math.count_nonzero(words_per_line,1,dtype=tf.int32)
            max_lines = tf.reduce_max(lines_per_doc)
            num_words = words_per_line[:,:max_lines]
            num_words = tf.reshape(num_words,(-1,))
            doc_input_reduced = docs[:,:max_lines,:max_words]
            
            #masks
            skip_lines = tf.not_equal(num_words,0)
            count_lines = tf.reduce_sum(tf.cast(skip_lines,tf.int32))
            mask_words = tf.sequence_mask(num_words,max_words)[skip_lines]    #batch*max_lines x max_words
            mask_words = tf.tile(tf.expand_dims(mask_words,1),[1,self.attention_heads,1])  #batch*max_lines x heads x max_words
            mask_lines = tf.sequence_mask(lines_per_doc,max_lines)    #batch x max_lines
            mask_lines = tf.tile(tf.expand_dims(mask_lines,1),[1,self.attention_heads,1])  #batch x heads x max_lines
                        
            #word embeddings
            doc_input_reduced = tf.reshape(doc_input_reduced,(-1,max_words))[skip_lines]
            word_embeds = self.embedding(doc_input_reduced)  #batch*max_lines x max_words x embed_dim
            word_embeds = self.word_drop(word_embeds,training=self.training)
            
            #word self attention
            word_q = self._split_heads(self.word_Q(word_embeds),count_lines)   #batch*max_lines x heads x max_words x depth
            word_k = self._split_heads(self.word_K(word_embeds),count_lines)   #batch*max_lines x heads x max_words x depth
            word_v = self._split_heads(self.word_V(word_embeds),count_lines)   #batch*max_lines x heads x max_words x depth
            word_self_out = self.word_self_att([word_q,word_v,word_k],[mask_words,mask_words],
                            training=self.training)
            
            #word target attention
            word_target = tf.tile(self.word_target,[count_lines,1,1,1])       #batch*max_lines x heads x 1 x depth
            word_targ_out = self.word_targ_att([word_target,word_self_out,word_self_out],[None,mask_words],
                            training=self.training)
            word_targ_out = tf.transpose(word_targ_out,perm=[0, 2, 1, 3])   #batch*max_lines x 1 x heads x depth
            line_embeds = tf.scatter_nd(tf.where(skip_lines),
                                        tf.reshape(word_targ_out,(count_lines,self.attention_size)),
                                        (batch_size*max_lines,self.attention_size))
            line_embeds = tf.reshape(line_embeds,(batch_size,max_lines,self.attention_size))
            line_embeds = self.line_drop(line_embeds,training=self.training)

            #line self attention
            line_q = self._split_heads(self.line_Q(line_embeds),batch_size)   #batch x heads x max_lines x depth
            line_k = self._split_heads(self.line_K(line_embeds),batch_size)   #batch x heads x max_lines x depth
            line_v = self._split_heads(self.line_V(line_embeds),batch_size)   #batch x heads x max_lines x depth
            line_self_out = self.line_self_att([line_q,line_v,line_k],[mask_lines,mask_lines],
                            training=self.training)
            
            #word target attention
            line_target = tf.tile(self.line_target,[batch_size,1,1,1])       #batch x heads x 1 x depth
            line_targ_out = self.line_targ_att([line_target,line_self_out,line_self_out],[None,mask_lines],
                            training=self.training)
            line_targ_out = tf.transpose(line_targ_out,perm=[0, 2, 1, 3])    #batch x 1 x heads x depth
            doc_embeds = tf.reshape(line_targ_out,(batch_size,self.attention_size))
            doc_embeds = self.doc_drop(doc_embeds,training=self.training)
            
            logits = self.classify(doc_embeds)
            return logits

        def _split_heads(self,x,batch_size):
            x = tf.reshape(x,(batch_size,-1,self.attention_heads,
                           int(self.attention_size/self.attention_heads)))
            return tf.transpose(x,perm=[0, 2, 1, 3])
            
    def __init__(self,embedding_matrix,num_classes,max_sents=201,max_words=15,
                 attention_heads=8,attention_size=400):
    
        self.ms = max_sents
        self.mw = max_words
        self.unk_tok = embedding_matrix.shape[0] - 1
        self.vocab_size = embedding_matrix.shape[0]
        self.model = self.hisan_model(embedding_matrix,num_classes,attention_size,attention_heads)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(0.0001,0.9,0.99,1e-08,False)

    def _list_to_numpy(self,inputval,noise=False):
    
        batch_size = len(inputval)
        retval = np.zeros((batch_size,self.ms,self.mw))
        for i,doc in enumerate(inputval):
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
        return retval

    @tf.function
    def _train_step(self,text,labels):
        with tf.GradientTape() as tape:
            predictions = self.model(text,training=True)
            loss = self.loss_object(labels,predictions)
        gradients = tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        return predictions, loss
        
    @tf.function
    def _predict_step(self,text):
        predictions = self.model(text,training=False)
        return predictions

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
        prevbest = 0
        pat_count = 0

        for ep in range(epochs):
        
            self.model.training = True

            #shuffle data
            xy = list(zip(data,labels))            
            random.shuffle(xy)
            data,labels = zip(*xy)
            data = np.array(data)
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

                #train step
                retval = self._list_to_numpy(data[start:stop],noise=True).astype(np.int32)
                predictions,loss = self._train_step(retval,labels[start:stop])

                #track correct predictions
                y_pred.extend(np.argmax(predictions,1))
                sys.stdout.write("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop,len(data),loss))
                sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
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

    def predict(self,data,batch_size=128):
    
        self.model.training = False
    
        y_pred = []
        for start in range(0,len(data),batch_size):

            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            retval = self._list_to_numpy(data[start:stop]).astype(np.int32)
            predictions = self._predict_step(retval)
            y_pred.append(np.argmax(predictions,1))

            sys.stdout.write("processed %i of %i records        \r" \
                             % (stop,len(data)))
            sys.stdout.flush()

        print()
        y_pred = np.concatenate(y_pred,0)
        return y_pred

    def score(self,data,labels,batch_size=128):  
        
        y_pred = self.predict(data,batch_size)
        micro = f1_score(labels,y_pred,average='micro')
        macro = f1_score(labels,y_pred,average='macro')
        return micro,macro
        
    def save(self,savepath):

        self.model.save_weights(savepath)

    def load(self,savepath):

        self.model.load_weights(savepath)


if __name__ == "__main__":

    '''
    dummy test data
    '''

    #params
    batch_size = 32
    epochs = 30
    train_samples = 10000
    test_samples = 1000
    vocab_size = 10000
    max_words = 1000
    num_classes = 3
    embedding_size = 300
    attention_heads = 8
    attention_size = 400
    
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
    model = hisan(vocab,num_classes,int(np.ceil(max_words/15)+1),15,
                  attention_heads,attention_size)
    model.train(X_train,y_train,batch_size,epochs,validation_data=(X_test,y_test),
                savebest=True,filepath='savedmodels/model.ckpt')
    model.load('savedmodels/model.ckpt')
