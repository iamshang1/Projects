import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#load posts
print "loading posts"
corpus = []
classes = []
for file in glob.glob(os.path.join('shakespeare/comedy/', '*.txt')):
    with open(file,'r') as f:
        text = f.read().replace("\n"," ").lower()
        corpus.append(text)
        classes.append(0)
for file in glob.glob(os.path.join('shakespeare/history/', '*.txt')):
    with open(file,'r') as f:
        text = f.read().replace("\n"," ").lower()
        corpus.append(text)
        classes.append(1)
for file in glob.glob(os.path.join('shakespeare/tragedy/', '*.txt')):
    with open(file,'r') as f:
        text = f.read().replace("\n"," ").lower()
        corpus.append(text)
        classes.append(2)

#vectorize posts
print "vectorizing posts"
vectorizer = TfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1,1), decode_error='ignore')
vector = vectorizer.fit_transform(corpus)
data = vector.toarray()

#plot data
print "plotting data"
pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)

colors = ['g','b','r']
for i in range(len(classes)):
    plt.scatter(pca_data[i:,0],pca_data[i:,1],s=100,c=colors[classes[i]])

green_patch = mpatches.Patch(color='g', label='comedy')
blue_patch = mpatches.Patch(color='b', label='tragedy')
red_patch = mpatches.Patch(color='r', label='history')
plt.legend(handles=[green_patch,blue_patch,red_patch],loc='upper left')
plt.title("PCA Visualization of Shakespeare Corpus")
plt.savefig('corpus.png')
plt.show()