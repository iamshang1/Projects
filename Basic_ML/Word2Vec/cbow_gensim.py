import numpy as np
from sklearn.datasets import fetch_20newsgroups
import unicodedata
import gensim
import string
import re
import collections
import logging
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load 20 newsgroups dataset
print "loading dataset"
dataset = fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes')).data
dataset = ' '.join(dataset)
dataset = unicodedata.normalize('NFKD', dataset).encode('ascii','ignore')

#convert dataset to list of sentences
print "converting dataset to list of sentences"
sentences = re.sub(r'-|\t|\n',' ',dataset)
sentences = sentences.split('.')
sentences = [sentence[2:].translate(None, string.punctuation).lower().split() for sentence in sentences]

#train word2vec
print "training word2vec"
model = gensim.models.Word2Vec(sentences, min_count=5, size=50, workers=4)

#get most common words
print "getting common words"
dataset = [item for sublist in sentences for item in sublist]
counts = collections.Counter(dataset).most_common(500)

#reduce embeddings to 2d using tsne
print "reducing embeddings to 2D"
embeddings = np.empty((500,50))
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

#save to disk
plt.savefig('plot.png')