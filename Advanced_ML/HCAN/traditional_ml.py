import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

#load saved files
print("loading data")
vocab = np.load('data/pubmed_embeddings.npy')
with open('data/pubmed_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
#convert each doc into a string
print("creating features and labels")
docs = []
labels = []
for key,value in data.items():
    docs.append(value['text'])
    labels.append(value['label'])
    
docstrings = []
for doc in docs:
    flattened = [word for line in doc for word in line]
    docstring = " ".join(flattened)
    docstrings.append(docstring)

#tfidf vectorization
vectorizer = TfidfVectorizer(min_df=5, stop_words='english',ngram_range=(1,2))
X = vectorizer.fit_transform(docstrings)

#label encoder
le = LabelEncoder()
y = le.fit_transform(labels)

#test train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,
                                random_state=1234,stratify=y)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.12,
                                  random_state=1234,stratify=y_train)

#classify using Naive Bayes
start_time = time.time()
print("training naive bayes")
clf = MultinomialNB()
clf.fit(X_train, y_train)
print("training time: %.4f" % (time.time() - start_time))

score = clf.score(X_test, y_test)
print("Naive Bayes accuracy: %.4f%%" % (score*100))

#classify using Logistic Regression
start_time = time.time()
print("training logistic regression")
clf = LogisticRegression('l1')
clf.fit(X_train, y_train)
print("training time: %.4f" % (time.time() - start_time))

score = clf.score(X_test, y_test)
print("logistic regression accuracy: %.4f%%" % (score*100))
