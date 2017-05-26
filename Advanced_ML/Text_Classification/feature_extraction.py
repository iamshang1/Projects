import sys
import ast
import re
from itertools import groupby
import numpy as np
import collections
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import logging
import pickle

#get json filepath
args = (sys.argv)
if len(args) != 2:
    raise Exception("Usage: python feature_extraction.py <path to Yelp json file>")
json_path = args[1]

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

#store records
labels = []
tokens = []
maxsentlen = 0
maxdoclen = 0

#process json one line at a time
with open(json_path,'r') as f:
    lineno = 0
    for line in f:
    
        lineno += 1
        sys.stdout.write("processing line %i of aprox 4.15 million     \r" \
                         % lineno)
        sys.stdout.flush()
        dic = ast.literal_eval(line)
        
        #only keep records from 2013 (to reduce dataset size)
        if dic['date'][:4]!='2013':
            continue
        
        text = dic['text']
                
        #process text
        text = text.lower()
        text = re.sub("'", '', text)
        text = re.sub("\.{2,}", '.', text)
        text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
        text = re.sub('\.', ' . ', text)
        text = re.sub('\?', ' ? ', text)
        text = re.sub('!', ' ! ', text)

        #tokenize
        text = text.split()
        
        #drop empty reviews
        if len(text) == 0:
            continue

        #split into sentences
        sentences = []
        sentence = []
        for t in text:
            if t not in ['.','!','?']:
                sentence.append(t)
            else:
                sentence.append(t)
                sentences.append(sentence)
                if len(sentence) > maxsentlen:
                    maxsentlen = len(sentence)
                sentence = []
        if len(sentence) > 0:
            sentences.append(sentence)
        
        #add split sentences to tokens
        tokens.append(sentences)
        if len(sentences) > maxdoclen:
            maxdoclen = len(sentences)
        
        #add label 
        labels.append(dic['stars'])
        
print '\nsaved %i records' % len(tokens)
        
#generate Word2Vec embeddings
print "generating word2vec embeddings"

#used all processed raw text to train word2vec
allsents = [sent for doc in tokens for sent in doc]

embedding_size = 200
model = Word2Vec(allsents, min_count=5, size=embedding_size, workers=4, iter=5)
model.init_sims(replace=True)
'''
#get most common words
print "getting common words"
allwords = [word for sent in allsents for word in sent]
counts = collections.Counter(allwords).most_common(500)

#reduce embeddings to 2d using tsne
print "reducing embeddings to 2D"
embeddings = np.empty((500,embedding_size))
for i in range(500):
    embeddings[i,:] = model[counts[i][0]]
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
embeddings = tsne.fit_transform(embeddings)

#plot embeddings
print "plotting most common words"
fig, ax = plt.subplots(figsize=(30, 30))
for i in range(500):
    ax.scatter(embeddings[i,0],embeddings[i,1])
    ax.annotate(counts[i][0], (embeddings[i,0],embeddings[i,1]))
plt.show()
'''
#save all word embeddings to matrix
print "saving word vectors to matrix"
vocab = np.zeros((len(model.wv.vocab)+1,embedding_size))
word2id = {}

#first row of embedding matrix isn't used so that 0 can be masked
for key,val in model.wv.vocab.iteritems():
    idx = val.__dict__['index'] + 1
    vocab[idx,:] = model[key]
    word2id[key] = idx
    
#normalize embeddings
vocab -= vocab.mean()
vocab /= vocab.std()

#reset first row to 0
vocab[0,:] = np.zeros((embedding_size))

#add additional word embedding for unknown words
vocab = np.concatenate((vocab, np.random.rand(1,embedding_size)))

#index for unknown words
unk = len(vocab)-1

#convert words to word indicies
print "converting words to indices"
data = {}
for idx,doc in enumerate(tokens):
    sys.stdout.write('processing %i of %i records       \r' % (idx+1,len(tokens)))
    sys.stdout.flush()
    dic = {}
    dic['label'] = labels[idx]
    dic['text'] = doc
    indicies = []
    for sent in doc:
        indicies.append([word2id[word] if word in word2id else unk for word in sent])
    dic['idx'] = indicies
    data[idx] = dic

#save preprocessed data and embeddings to disk
print "\nsaving data to disk"
np.save('embeddings',vocab)
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
