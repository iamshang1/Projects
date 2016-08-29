import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk.stem

pos = pd.read_csv('positive.gz',sep=',',header=0)
neg = pd.read_csv('negative.gz',sep=',',header=0)

pos_text = pos['Summary'] + " " + pos['Text']
neg_text = neg['Summary'] + " " + neg['Text']
pos_text = pos_text.map(lambda x: x.decode('utf8', 'ignore').replace('<br />',' '))
neg_text = neg_text.map(lambda x: x.decode('utf8', 'ignore').replace('<br />',' '))

pos_train = pos_text.iloc[:40000]
neg_train = neg_text.iloc[:40000]
pos_test = pos_text.iloc[40000:]
neg_test = neg_text.iloc[40000:]

X_train = pos_train.append(neg_train)

y_train = np.append(np.ones((len(pos_train))),np.zeros((len(neg_train))))
y_test_pos = np.ones((len(pos_test)))
y_test_neg = np.zeros((len(neg_test)))

print "vectorizing reviews"
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer = StemmedTfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1, 1), decode_error='ignore')
X_train = vectorizer.fit_transform(X_train)
X_test_pos = vectorizer.transform(pos_test)
X_test_neg = vectorizer.transform(neg_test)

#classify emails with naive bayes
print "classifing reviews"
clf = MultinomialNB()
clf.fit(X_train, y_train)
score_pos = clf.score(X_test_pos, y_test_pos)
score_neg = clf.score(X_test_neg, y_test_neg)

print "Naive Bayes accuracy on %i positive reviews: %.2f%%" % (len(y_test_pos), score_pos*100)
print "Naive Bayes accuracy on %i negative reviews: %.2f%%" % (len(y_test_neg), score_neg*100)
