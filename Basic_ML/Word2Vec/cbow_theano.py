from sklearn.datasets import fetch_20newsgroups
import unicodedata
import numpy as np
import theano
import theano.tensor as T
import string
import re
import sys
import collections
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

#load 20 newsgroups dataset
print "loading dataset"
dataset = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')).data
dataset = ' '.join(dataset)
dataset = unicodedata.normalize('NFKD', dataset).encode('ascii','ignore')
    
#cbow word2vec model
class cbow(object):
    '''
    word2vec model using continuous bag of words
    
    parameters:
      - dataset: string
        text corpus on which word2vec will be applied
      - dictionary_size: int (default 50000)
        number of unique words to vectorize, ranked by most common
      - embedding_size: int (default 50)
        length of embedding vector for each vectorized word
      - skip_window: int (default 5)
        number of words to left and right of target word to consider
      - learning_rate: float (default 0.1)
        learning rate used for gradient descent
      - negative_sampling: float (default 25)
        number of words to use for negative sampling
        
    methods:
      - train(iterations)
        train word2vec model on all words in dataset
        parameters:
          - iterations: int (default 1)
            number of times to iterate over all words in dataset
      - visualize_most_common(words)
        saves scatter plot of most common words to disk
        parameters:
          - words: int (default 500)
            number of words to visualize, ranked by most common
      - get_embedding(word)
        get embedding vector for given word
        parameters:
          - word: int or string
            wordid or word for which to retrieve embedding
        outputs:
          - numpy array representing word embedding
    '''
    def __init__(self,dataset,dictionary_size=50000,embedding_size=50,skip_window=5,learning_rate=0.1,negative_sampling=25):
        self.ds = dictionary_size
        self.es = embedding_size
        self.sw = skip_window
        self.lr = learning_rate
        self.ns = negative_sampling
        self._tokenize(dataset)
        
        #nn architecture
        self.input = T.matrix()
        self.w1 = theano.shared((np.random.rand(self.ds,self.es).astype(theano.config.floatX)-0.5),borrow=True)
        self.activeidx = T.ivector()
        self.activew1 = T.take(self.w1, self.activeidx, axis=0)
        self.l1out = T.dot(self.input,self.activew1)
        self.sampidx = T.ivector()
        self.sampw2 = T.take(self.w1.T, self.sampidx, axis=1)
        self.l2out = T.nnet.softmax(T.dot(self.l1out,self.sampw2))
        self.target = T.matrix()
       
        #nn functions
        self.params = [self.w1]
        self.cost = T.nnet.categorical_crossentropy(self.l2out,self.target).mean()
        self.gparams = [T.grad(self.cost, param) for param in self.params]
        self.propogate = theano.function([self.input,self.target,self.activeidx,self.sampidx],self.cost,\
            updates=[(param,param-self.lr*gparam) for param,gparam in zip(self.params,self.gparams)],allow_input_downcast=True)
        
    def _tokenize(self,dataset):
        '''
        lowercase all strings and remove punctuation
        tokenize dataset
        convert words to unique ids
        '''
        print 'tokenizing dataset'          
        #lowercase and remove tabs, linebreaks, and punctuation
        tokens = re.sub(r'-|\t|\n',' ',dataset)
        tokens = tokens.translate(None, string.punctuation).lower().split()

        #map most common tokens to ids
        print 'generating word ids'
        self.most_common = [['UNK', -1]]
        self.most_common.extend(collections.Counter(tokens).most_common(self.ds-1))
        counts = collections.Counter(tokens)

        #dictionary to store mappings between ids and words
        self.word2ids = dict()
        for word, count in self.most_common:
            self.word2ids[word] = len(self.word2ids)
        self.ids2word = dict(zip(self.word2ids.values(), self.word2ids.keys()))

        #convert tokenized dataset from words to ids
        print 'converting words to ids in dataset'
        self.data = list()
        unk_count = 0
        for word in tokens:
            
            #remove words that appear less than 5 times
            if counts[word] < 5:
                continue
            if word in self.word2ids:
                index = self.word2ids[word]
            else:
                index = 0
                unk_count += 1
            self.data.append(index)
        self.most_common[0][1] = unk_count
        del tokens
        
        print "total words in dataset: %i" % len(self.data)
        
    def train(self,iterations=1):
        '''
        train word2vec model on all words in dataset
        parameters:
          - iterations: int (default 1)
            number of times to iterate over all words in dataset
        '''
        for i in range(iterations):
            print "training iteration %i            " % (i+1)
            for idx in range(self.sw,len(self.data)-self.sw):
                sys.stdout.write("processing word %i of %i    \r" % (idx+1, len(self.data)-5))
                sys.stdout.flush()
                
                #get word id
                wordid = self.data[idx]
                
                #probabilistically discard common words, skip unknown words
                count = self.most_common[wordid][1]
                if wordid==0 or np.random.rand() < np.sqrt(0.00005/count):
                    continue
                
                #train
                self._train_one_step(idx)
        
    def _train_one_step(self,idx):
        '''
        train one step of word2vec model on dataset
        '''
        #init arrays to store neighbor words
        batch = np.zeros((1,self.ds))
        labels = np.zeros((1,self.ds))
        
        #indices for neighbor words and negative sampling
        activeidx = []
        sampidx = []
        wordid = self.data[idx]
        
        #get neighboring words
        for j in range(-self.sw,self.sw+1):
            if j != 0:
                wordid = self.data[idx+j]
                batch[0,wordid] += 1./(self.sw*2)
                activeidx.append(wordid)
                
        wordid = self.data[idx]
        labels[0,wordid] += 1
        
        #include target word id in negative sampling
        sampidx.append(wordid)
        
        #include random additional words in negative sampling
        numbers = range(self.ds)
        np.random.shuffle(numbers)
        numbers = numbers[:self.ns]
        sampidx.extend(numbers)
        
        #negative sampling filter for target distribution
        labels = labels[:,sampidx]
        sampidx = np.array(sampidx)
        
        #filter input to only include active neighbor words
        activeidx = list(set(activeidx))
        batch = batch[:,activeidx]
        activeidx = np.array(activeidx)
        
        return self.propogate(batch,labels,activeidx,sampidx)
            
    def visualize_most_common(self,words=500):
        '''
        saves scatter plot of most common words to disk
        parameters:
          - words: int (default 500)
            number of words to visualize, ranked by most common
        '''
        #get most common words
        embeddings = np.empty((words,self.es))
        for id in range(words):
        
            #skip id 0 since it represents all unknown words
            embeddings[id,:] = self.w2.get_value()[:,id+1]
            
        #reduce embeddings to 2d using tsne
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
        embeddings = tsne.fit_transform(embeddings)
        
        #plot
        fig, ax = plt.subplots(figsize=(30, 30))
        for id in range(words):
            ax.scatter(embeddings[id,0],embeddings[id,1])
            ax.annotate(self.ids2word[id+1], (embeddings[id,0],embeddings[id,1]))
        
        #save to disk
        plt.savefig('cbow.png')
            
    def get_embedding(self,word):
        '''
        get embedding vector for given word
        parameters:
          - word: int or string
            wordid or word for which to retrieve embedding
        outputs:
          - numpy array representing word embedding
        '''
        if type(word) == int:
            id = word
        elif type(word) == str:
            if not word in self.word2ids:
                raise Exception('word not found in dictionary')
            id = self.word2ids[word]
        else:
            raise Exception('word must be string or id')
        return self.w1.get_value()[:,id]

#build and train model        
cbow = cbow(dataset)
cbow.train()
cbow.visualize_most_common()