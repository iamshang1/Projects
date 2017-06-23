import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sys
import operator
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class hierarchical_attention_network(object):
    '''
    hierarchical attention network for document classification
    https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num classes: int
        number of output classes
      - max_sents: int
        maximum number of sentences per document
      - max_words: int
        maximum number of words per sentence
      - rnn_type: string (default: "gru")
        rnn cells to use, can be "gru" or "lstm"
      - rnn_units: int (default: 100)
        number of rnn units to use for embedding layers
      - attention_context: int (default: 50)
        number of dimensions to use for attention context layer
      - dropout_keep: float (default: 0.5)
        dropout keep rate for final softmax layer
       
    methods:
      - pretrain(data,epochs=5)
        pretrain network on unlabeled data by predicting tfidf vector associated with each doc
      - train(data,labels,epochs=30,savebest=False,filepath=None)
        train network on given data
      - predict(data)
        return the one-hot-encoded predicted labels for given data
      - predict_proba(data)
        return the probability of each class for given data
      - score(data,labels,bootstrap=False,bs_samples=100)
        return the accuracy of predicted labels on given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
      - visualize_word_importance(self, document, word2id, fname)
        visualize word importance of given document
      - most_important_words(data, word2id)
        get most important words for classification task
      - plot_doc_embeddings(data,labels,fname,key=None)
        visualize document embeddings created by network
    '''
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,rnn_type="gru",
                 rnn_units=200,attention_context=300,dropout_keep=0.5):

        self.rnn_units = rnn_units
        if rnn_type == "gru":
            self.rnn_cell = GRUCell
        elif rnn_type == "lstm":
            self.rnn_cell = LSTMCell
        else:
            raise Exception("rnn_type parameter must be set to gru or lstm")
        self.dropout_keep = dropout_keep
        self.vocab = embedding_matrix
        self.embedding_size = embedding_matrix.shape[1]

        #shared variables
        with tf.variable_scope('words'):
            self.word_atten_W = tf.Variable(self._ortho_weight(2*rnn_units,attention_context),name='word_atten_W')
            self.word_atten_b = tf.Variable(np.asarray(np.zeros(attention_context),dtype=np.float32),name='word_atten_b')
            self.word_softmax = tf.Variable(self._ortho_weight(attention_context,1),name='word_softmax')
        with tf.variable_scope('sentence'):
            self.sent_atten_W = tf.Variable(self._ortho_weight(2*rnn_units,attention_context),name='sent_atten_W')
            self.sent_atten_b = tf.Variable(np.asarray(np.zeros(attention_context),dtype=np.float32),name='sent_atten_b')
            self.sent_softmax = tf.Variable(self._ortho_weight(attention_context,1),name='sent_softmax')
        with tf.variable_scope('classify'):
            self.W_softmax = tf.Variable(self._ortho_weight(rnn_units*2,num_classes),name='W_softmax')
            self.b_softmax = tf.Variable(np.asarray(np.zeros(num_classes),dtype=np.float32),name='b_softmax')
        with tf.variable_scope('pretrain'):
            self.W_pretrain = tf.Variable(self._ortho_weight(rnn_units*2,self.embedding_size),name='W_pretrain')
            self.b_pretrain = tf.Variable(np.asarray(np.zeros(self.embedding_size),dtype=np.float32),name='b_pretrain')
        
        #word embeddings
        with tf.variable_scope('embeddings'):
            self.embeddings = tf.cast(tf.Variable(embedding_matrix,name='embeddings'),tf.float32)
        self.dropout = tf.placeholder(tf.float32)
        
        #sentence input and mask
        self.sent_input = tf.placeholder(tf.int32, shape=[max_words])
        self.word_mask = tf.not_equal(self.sent_input,tf.zeros_like(self.sent_input))
        self.word_nonzero = tf.boolean_mask(self.sent_input,self.word_mask)
        self.word_embeds = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings,self.word_nonzero),0)
        self.sen_len = self._length(self.word_embeds)
        with tf.variable_scope('words'):
            [self.word_outputs_fw,self.word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    self.word_embeds,sequence_length=self.sen_len,dtype=tf.float32)
        self.word_outputs = tf.concat((tf.squeeze(self.word_outputs_fw,[0]),tf.squeeze(self.word_outputs_bw,[0])),1)
        self.word_u = tf.nn.tanh(tf.matmul(self.word_outputs,self.word_atten_W) + self.word_atten_b)
        self.word_exp = tf.exp(tf.matmul(self.word_u,self.word_softmax))
        self.word_atten = self.word_exp/tf.reduce_sum(self.word_exp)

        #doc input and mask
        self.doc_input = tf.placeholder(tf.int32, shape=[max_sents,max_words])
        self.sent_sum = tf.reduce_sum(self.doc_input,1)
        self.sent_mask = tf.not_equal(self.sent_sum,tf.zeros_like(self.sent_sum))
        self.sent_nonzero = tf.boolean_mask(self.doc_input,self.sent_mask)
        self.doc_len = self._length(tf.expand_dims(self.sent_nonzero,0))
        
        #get sent embeddings
        self.sent_embeds = tf.map_fn(self._sent_embedding_step,self.sent_nonzero,dtype=tf.float32)
        self.sent_embeds = tf.reshape(self.sent_embeds,(1,self.doc_len[0],rnn_units*2))

        #sentence rnn layer
        with tf.variable_scope('sentence'):
            [self.sent_outputs_fw,self.sent_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    self.sent_embeds,sequence_length=self.doc_len,dtype=tf.float32)
        self.sent_outputs = tf.concat((tf.squeeze(self.sent_outputs_fw,[0]),tf.squeeze(self.sent_outputs_bw,[0])),1)
        
        #sentence attention
        self.sent_u = tf.nn.tanh(tf.matmul(self.sent_outputs,self.sent_atten_W) + self.sent_atten_b)
        self.sent_exp = tf.exp(tf.matmul(self.sent_u,self.sent_softmax))
        self.sent_atten = self.sent_exp/tf.reduce_sum(self.sent_exp)
        self.doc_embed = tf.transpose(tf.matmul(tf.transpose(self.sent_outputs),self.sent_atten))
        self.doc_embed_drop = tf.nn.dropout(self.doc_embed,self.dropout)

        #pretraining functions
        self.pred_embed = tf.nn.tanh(tf.matmul(self.doc_embed_drop,self.W_pretrain)+self.b_pretrain)
        self.target_embed = tf.placeholder(tf.float32, shape=[self.embedding_size])
        self.pretrain_loss = tf.reduce_mean(tf.squared_difference(self.pred_embed,self.target_embed))
        self.word_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"words")
        self.sent_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"sentence")
        self.pretrain_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"pretrain")
        self.pretrain_optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(
                                  self.pretrain_loss,var_list=self.word_vars+self.sent_vars+self.pretrain_vars)

        #classification functions
        self.output = tf.matmul(self.doc_embed_drop,self.W_softmax)+self.b_softmax
        self.prediction = tf.nn.softmax(self.output)
        
        #loss, accuracy, and training functions
        self.labels = tf.placeholder(tf.float32, shape=[num_classes])
        self.labels_rs = tf.expand_dims(self.labels,0)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output,labels=self.labels_rs))
        self.optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.999).minimize(self.loss)

        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
    
    def _length(self,sequence):
        '''
        return length of dynamic input tensor for rnn
        '''
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
    
    def _sent_embedding_step(self,sent):
        '''
        get sentence embeddings
        '''
        word_mask = tf.not_equal(sent,tf.zeros_like(sent))
        word_nonzero = tf.boolean_mask(sent,word_mask)
        word_embeds = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings,word_nonzero),0)
        with tf.variable_scope('words',reuse=True):
            [word_outputs_fw,word_outputs_bw],_ = \
                    tf.nn.bidirectional_dynamic_rnn(
                    self.rnn_cell(self.rnn_units),self.rnn_cell(self.rnn_units),
                    word_embeds,sequence_length=self._length(word_embeds),dtype=tf.float32)
        word_outputs = tf.concat((tf.squeeze(word_outputs_fw,[0]),tf.squeeze(word_outputs_bw,[0])),1)
        
        #attention
        word_u = tf.nn.tanh(tf.matmul(word_outputs,self.word_atten_W) + self.word_atten_b)
        word_exp = tf.exp(tf.matmul(word_u,self.word_softmax))
        word_atten = word_exp/tf.reduce_sum(word_exp)
        sent_embed = tf.matmul(tf.transpose(word_outputs),word_atten)
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
    
    def _list_to_numpy(self,inputval):
        '''
        convert variable length lists of input values to zero padded numpy array
        '''
        if type(inputval) == list:
            retval = np.zeros((self.max_sents,self.max_words))
            for i,line in enumerate(inputval):
                for j, word in enumerate(line):
                    retval[i,j] = word
            return retval
        elif type(inputval) == np.array:
            return inputval
        else:
            raise Exception("invalid input type")
    
    def pretrain(self,data,epochs=5):
        '''
        pretrain network on unlabeled data by predicting tfidf vector associated with each doc
                
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - epochs: int (default: 5)
            number of epochs to train for
        
        outputs:
            None
        '''
        #flatten documents
        print "flattening texts"
        texts = []
        for i in range(len(data)):
            if type(data[i]) == np.array:
                text = data[i].flatten()
                text = " ".join([str(int(idx)) for idx in text if idx != 0])
            elif type(data[i]) == list:
                text = [idx for line in data[i] for idx in line]
                text = " ".join([str(int(idx)) for idx in text if idx != 0])
            else:
                raise Exception("invalid input type")
            texts.append(text)
            
        #run tfidf
        print "creating tfidf vectors"
        tfidf_vocab = {str(int(i)):i for i in range(len(self.vocab))}
        model = TfidfVectorizer(texts,ngram_range=(1,1),vocabulary=tfidf_vocab)
        tfidf = model.fit_transform(texts)
        tfidf = tfidf.toarray()
        
        #normalize
        tfidf = tfidf/np.expand_dims(np.sum(tfidf,1),1)
        
        #tfidf-weighted word embeddings
        weighted_embs = []
        for doc in range(tfidf.shape[0]):
            emb = np.zeros(self.embedding_size)
            for word in range(tfidf.shape[1]):
                emb += self.vocab[word]*tfidf[doc,word]
            weighted_embs.append(emb)
        weighted_embs = np.array(weighted_embs)
        
        #normalize
        weighted_embs = (weighted_embs - weighted_embs.mean())/(weighted_embs.std()*2.5)
        weighted_embs = np.clip(weighted_embs,-1,1)
        
        #train
        for ep in range(epochs):
            for doc in range(len(data)):
       
                inputval = self._list_to_numpy(data[doc])
                feed_dict = {self.doc_input:inputval,self.target_embed:weighted_embs[doc],self.dropout:self.dropout_keep}
                cost,_ = self.sess.run([self.pretrain_loss,self.pretrain_optimizer],feed_dict=feed_dict)

                sys.stdout.write("pretrain epoch %i, document %i of %i, loss: %f      \r"\
                             % (ep+1,doc+1,len(data),cost))
                sys.stdout.flush()
            print ""
    
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
            print ""
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
        
    def score(self,data,labels,bootstrap=False,bs_samples=100):
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
                correct += 1.
        
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
        
    def visualize_word_importance(self, document, word2id, fname):
        '''
        visualize word importance of given document
        
        parameters:
          - document: numpy array
            2d numpy array (sentence x word ids) representing document
          - word2id: dictionary
            dictionary of words to their corresponding word2vec indices
          - fname: string
            path to save word importance image
        
        outputs:
            None
        '''
        unk = len(word2id)+1
        id2word = dict(zip(word2id.values(), word2id.keys()))
        
        #get line importance
        document = self._list_to_numpy(document)
        feed_dict = {self.doc_input:document,self.dropout:1.0}
        line_importance = self.sess.run(self.sent_atten,feed_dict=feed_dict)
        
        #get word importance for each line
        word_importance = []
        for line in document:
            #skip empty lines
            if np.sum(line) == 0:
                continue
            feed_dict = {self.sent_input:line,self.dropout:1.0}
            word_importance.append(self.sess.run(self.word_atten,feed_dict=feed_dict))
        
        #get size of lines and sentences in document
        maxdoclen = 0
        maxsentlen = 0
        for i in range(len(document)):
            start = 0
            if np.sum(document[i]) == 0:
                continue
            maxdoclen += 1
            for j in range(len(document[i])):
                if document[i][j] == 0:
                    continue
                elif document[i][j] == unk:
                    word = '<unk>'
                else:
                    word = id2word[document[i][j]]
                    start += len(word)
                    if start > maxsentlen:
                        maxsentlen = start
                        
        #plot lines and words shaded by importance
        fig = plt.figure(figsize=((maxsentlen)/3,(maxdoclen)/3))
        ax = fig.add_subplot(111)
        lineptr = 0
        for line in document:
            wordptr = 0
            start = 0
            if np.sum(line) == 0:
                continue
            for wordidx in line:
                if wordidx == 0:
                    continue
                elif wordidx == unk:
                    word = '<unk>'
                else:
                    word = id2word[wordidx]
                alpha = line_importance[lineptr]*0.7
                ax.text(0,maxdoclen-lineptr,'line %i' % (lineptr+1),
                        bbox={'facecolor':'red','alpha':alpha,'pad':3})
                alpha = word_importance[lineptr][wordptr]*0.7
                ax.text(start+5,maxdoclen-lineptr,word,
                        bbox={'facecolor':'blue','alpha':alpha,'pad':3})
                start += int(float(len(word))/2.5+1.)
                wordptr += 1
            lineptr += 1
        ax.axis([0,maxsentlen+2,0,maxdoclen+2])
        ax.set_axis_off()
        
        #save to disk and reset
        fig.savefig(fname,bbox_inches='tight')
        plt.close(fig)
        
    def most_important_words(self, data, word2id):
        '''
        get most important words for classification task
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - word2id: dictionary
            dictionary of words to their corresponding word2vec indices
        
        outputs: dictionary with following structure
          - key: index of class
          - value: list of tuples of words and their associated weights for that
                   class, ranked by importance
        '''
        #dictionary to store word weights
        weight_dics = {}
        
        for i,doc in enumerate(data):
            sys.stdout.write("processing document %i of %i      \r"\
                                 % (i+1,len(data)))
            sys.stdout.flush()

            unk = len(word2id)+1
            id2word = dict(zip(word2id.values(), word2id.keys()))
            
            #get line importance
            doc = self._list_to_numpy(doc)
            feed_dict = {self.doc_input:doc,self.dropout:1.0}
            line_importance = self.sess.run(self.sent_atten,feed_dict=feed_dict)
            
            #get word importance for each line
            idx = 0
            word_importance = []
            flattened_idx = []
            for line in doc:
                #skip empty lines
                if np.sum(line) == 0:
                    continue
                feed_dict = {self.sent_input:line,self.dropout:1.0}
                word_imp_vec = self.sess.run(self.word_atten,feed_dict=feed_dict)
                word_imp_vec = word_imp_vec * line_importance[idx]
                word_importance.append(list(word_imp_vec))
                idx += 1
                for word in line:
                    if word != 0:
                        flattened_idx.append(word)
            
            #flatten doc
            flattened_idx = np.array(flattened_idx)
            flattened_imp = np.array([imp for line in word_importance for imp in line])
            
            #order in terms of highest weight
            order = flattened_imp.argsort()[::-1]
            ranked_idx = flattened_idx[order]
            ranked_imp = flattened_imp[order]
            
            #keep highest weight per word
            vocab = []
            final_words = []
            final_imp = []
            for idx in range(len(ranked_idx)):
                if ranked_idx[idx] in vocab:
                    continue
                else:
                    final_words.append(id2word[ranked_idx[idx]] if ranked_idx[idx] in id2word else "<unk>")
                    final_imp.append(ranked_imp[idx])
                    vocab.append(ranked_idx[idx])
            
            #update weight dictionary
            for idx,word in enumerate(final_words):
                if word in weight_dics:
                    weight_dics[word].append(final_imp[idx])
                else:
                    weight_dics[word] = [final_imp[idx]]
        
        print ''
                    
        #get mean word weight 
        for key,val in weight_dics.iteritems():
            weight_dics[key] = np.mean(val)
        
        #sort words by importance
        word_importance = sorted(weight_dics.items(),key=operator.itemgetter(1))[::-1]
        
        return word_importance
        
    def plot_doc_embeddings(self,data,labels,fname,key=None):
        '''
        visualize document embeddings created by network
        
        parameters:
          - data: numpy array
            3d numpy array (doc x sentence x word ids) of input data
          - labels: numpy array
            2d numpy array of one-hot-encoded labels
          - fname: string
            path to save doc embedding image
          - key: list (optional)
            list of strings representing name of each class
        
        outputs:
            None
        '''
        embeds = []
        labels = np.argmax(labels,1)
        
        #get document embeddings
        for i,doc in enumerate(data):
            sys.stdout.write("processing document %i of %i      \r"\
                                 % (i+1,len(data)))
            sys.stdout.flush()
            
            doc = self._list_to_numpy(doc)
            feed_dict = {self.doc_input:doc,self.dropout:1.0}
            doc_embed = self.sess.run(self.doc_embed,feed_dict=feed_dict)
            embeds.append(doc_embed.flatten())
            
        embeds = np.array(embeds)
        
        #reduce dimensionality
        pca = PCA(n_components=2)
        embeds = pca.fit_transform(embeds)
        
        #plot embeddings
        colors = ['r','g','b','c','m','y','orange','k','grey']
        for i in range(len(embeds)):
            plt.scatter(embeds[i,0],embeds[i,1],s=25,c=colors[labels[i]])
            
        #add legend
        if isinstance(key,collections.Iterable):
            patches = []
            for k in range(len(key)):
                patches.append(mpatches.Patch(color=colors[k],label=key[k]))
            plt.legend(handles=patches,loc='upper left',bbox_to_anchor=(1.0, 1.0))
        plt.title("HAN Document Embeddings by Class")
        plt.tick_params(axis='both',which='both',bottom='off',left='off',labelbottom='off',labelleft='off')
        
        #save
        plt.savefig(fname,bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()
        
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
    print "building hierarchical attention network"
    nn = hierarchical_attention_network(vocab,classes,max_sents,max_words)
    
    if not os.path.exists('./savedmodels'):
        os.makedirs('./savedmodels')
    nn.train(X_train,y_train,epochs=5,validation_data=(X_test,y_test),
             savebest=True,filepath='./savedmodels/han.ckpt')
    
    #load best nn
    nn.load('./savedmodels/han.ckpt')
    acc = nn.score(X_test,y_test)
    y_pred = np.argmax(nn.predict(X_test),1)
    print "HAN - test set accuracy: %.4f" % (acc*100)
