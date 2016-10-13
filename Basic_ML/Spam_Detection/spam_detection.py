import tarfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
import nltk.stem

#load emails
print "loading emails"
ham_gzip = tarfile.open('ham.tar.gz', 'r:gz')
ham = [ham_gzip.extractfile(f).read() for f in ham_gzip.getnames()]
ham_gzip.close()
spam_gzip = tarfile.open('spam.tar.gz', 'r:gz')
spam = [spam_gzip.extractfile(f).read() for f in spam_gzip.getnames()]
spam_gzip.close()

y_ham = np.zeros((len(ham)))
y_spam = np.ones((len(spam)))
y = np.concatenate((y_ham, y_spam))
data = ham + spam

#vectorize emails
print "vectorizing emails"
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
vectorizer = StemmedTfidfVectorizer(min_df=3, stop_words='english', ngram_range=(1, 2), decode_error='ignore')
X = vectorizer.fit_transform(data)

#classify emails with naive bayes
print "classifing emails"
clf = MultinomialNB()

accuracy = []
precision = []
recall = []
f1score = []

#crossvalidation
cv = StratifiedKFold(y, n_folds=3, shuffle=True)
for train, test in cv:
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = (clf.score(X_test, y_test))
    accuracy.append(score)
    pre, rec, f1, support = precision_recall_fscore_support(y_test, pred)
    precision.append(pre[0])
    recall.append(rec[0])
    f1score.append(f1[0])
    print "accuracy: %.2f%%, precision: %.4f, recall: %.4f, f1 score: %.4f" % (score*100, pre[0], rec[0], f1[0])

print "\noverall accuracy: %.2f%%\noverall precision: %.4f\noverall recall: %.4f\noverall f1 score: %.4f\n" % (np.mean(accuracy)*100, np.mean(precision), np.mean(recall), np.mean(f1score))

#get overall feature probabilities
clf.fit(X, y)
log_prob = clf.feature_log_prob_
log_prob_ham = log_prob[0,:]
log_prob_spam = log_prob[1,:]
ham_words = log_prob_ham.argsort()[-25:][::-1]
spam_words = log_prob_spam.argsort()[-25:][::-1]
words = vectorizer.get_feature_names()
ham_words = [str(words[i]) for i in ham_words]
spam_words = [str(words[i]) for i in spam_words]
print "words most likely to indicate ham"
print ham_words,"\n"
print "words most likely to indicate spam"
print spam_words,"\n"