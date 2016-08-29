import pandas as pd
import numpy as np
import csv, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import nltk.stem
import nltk
import sys

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
print "classifing reviews w/ Naive Bayes"
clf = MultinomialNB()
clf.fit(X_train, y_train)
nb_proba_train = clf.predict_proba(X_train)[:,1][:,np.newaxis]
nb_proba_pos = clf.predict_proba(X_test_pos)[:,1][:,np.newaxis]
nb_proba_neg = clf.predict_proba(X_test_neg)[:,1][:,np.newaxis]

def load_sentiwordnet():
    print 'loading sentiwordnet'
    sent_scores = collections.defaultdict(list)
    with open("SentiWordNet_3.0.0_20130122.txt", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t',quotechar='"')
        for line in reader:
            if line[0].startswith("#"):
                continue
            if len(line)==1:
                continue
            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line
            if len(POS)==0 or len(ID)==0:
                continue
            for term in SynsetTerms.split(" "):
                term = term.split("#")[0]
                term = term.replace("-", " ").replace("_", " ")
                key = "%s/%s"%(POS, term.split("#")[0])
                sent_scores[key].append((float(PosScore),float(NegScore)))
    for key, value in sent_scores.items():
        sent_scores[key] = np.mean(value, axis=0)
    return sent_scores

def evaluate_sentiment(text):
    pos_score = 0
    neg_score = 0
    tokened = nltk.word_tokenize(text)
    pos_pairs = nltk.pos_tag(tokened)
    for tuple in pos_pairs:
        pos = ''
        if tuple[1] == "NN":
            pos = 'n/'
        if tuple[1] == "JJ":
            pos = 'a/'
        if tuple[1] == "VB":
            pos = 'v/'
        if tuple[1] == "RB":
            pos = 'r/'
        try:
            pos_score += sentiwordnet[pos+tuple[0].lower()][0]
            neg_score += sentiwordnet[pos+tuple[0].lower()][1]
        except:
            pass
    return pos_score, neg_score
    
sentiwordnet = load_sentiwordnet()
X_train = pos_train.append(neg_train)

swn_proba_train = np.zeros((len(X_train),1))
processed = 0
for i in range(len(X_train)):
    pos_score,neg_score = evaluate_sentiment(X_train.iloc[i])
    if pos_score == 0 and neg_score == 0:
        swn_proba_train[i,0] += 0.5
    else:
        swn_proba_train[i,0] += pos_score/(pos_score + neg_score)
    processed += 1
    sys.stdout.write('SentiWordNet processed %i of %i training set reviews \r' % (processed, len(X_train)))
    sys.stdout.flush()
print ''
    
swn_proba_pos = np.zeros((len(pos_test),1))
processed = 0
for i in range(len(pos_test)):
    pos_score,neg_score = evaluate_sentiment(pos_test.iloc[i])
    if pos_score == 0 and neg_score == 0:
        swn_proba_pos[i,0] += 0.5
    else:
        swn_proba_pos[i,0] += pos_score/(pos_score + neg_score)
    processed += 1
    sys.stdout.write('SentiWordNet processed %i of %i positive test set reviews \r' % (processed, len(pos_test)))
    sys.stdout.flush()
print ''
    
swn_proba_neg = np.zeros((len(neg_test),1))
processed = 0
for i in range(len(neg_test)):
    pos_score,neg_score = evaluate_sentiment(neg_test.iloc[i])
    if pos_score == 0 and neg_score == 0:
        swn_proba_neg[i,0] += 0.5
    else:
        swn_proba_neg[i,0] += pos_score/(pos_score + neg_score)
    processed += 1
    sys.stdout.write('SentiWordNet processed %i of %i negative test set reviews \r' % (processed, len(neg_test)))
    sys.stdout.flush()
print ''
    
print "training logistic regression classifier"
comb_train = np.concatenate((swn_proba_train,nb_proba_train),1)
comb_test_pos = np.concatenate((swn_proba_pos,nb_proba_pos),1)
comb_test_neg = np.concatenate((swn_proba_neg,nb_proba_neg),1)
lr = LogisticRegression()
lr.fit(comb_train, y_train)
score_pos = lr.score(comb_test_pos, y_test_pos)
score_neg = lr.score(comb_test_neg, y_test_neg)

print "Combined accuracy on %i positive reviews: %.2f%%" % (len(y_test_pos), score_pos*100)
print "Combined accuracy on %i negative reviews: %.2f%%" % (len(y_test_neg), score_neg*100)
