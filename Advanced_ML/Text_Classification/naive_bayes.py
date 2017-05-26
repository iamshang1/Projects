import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

#load saved files
print "loading data"
vocab = np.load('embeddings.npy')
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    
#convert each doc into a string
print "creating features and labels"
docs = []
labels = []
for key,value in data.iteritems():
    docs.append(value['text'])
    labels.append(value['label'])
    
docstrings = []
for doc in docs:
    flattened = [word for line in doc for word in line]
    docstring = " ".join(flattened)
    docstrings.append(docstring)

#tfidf vectorization
vectorizer = TfidfVectorizer(min_df=3, stop_words='english',ngram_range=(1, 2))
X = vectorizer.fit_transform(docstrings)

#label encoder
le = LabelEncoder()
y = le.fit_transform(labels)

#kfold cross validation
splits = 10
kf = StratifiedKFold(n_splits=splits,shuffle=True,random_state=1234)

#classify using Naive Bayes
print "training naive bayes"
scores = []
i = 0
for train_index, test_index in kf.split(X,y):
    i += 1

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    scores.append(score)

    print "Naive Bayes - kfold %i of %i accuracy: %.4f%%" % (i,splits,score*100)
    
print "Naive Bayes - overall accuracy: %.4f" % (np.mean(scores)*100)
