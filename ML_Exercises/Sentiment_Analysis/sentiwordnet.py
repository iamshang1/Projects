import csv, collections
import pandas as pd
import numpy as np
import nltk
import sys

pos = pd.read_csv('positive.gz',sep=',',header=0)
neg = pd.read_csv('negative.gz',sep=',',header=0)

pos_text = pos['Summary'] + " " + pos['Text']
neg_text = neg['Summary'] + " " + neg['Text']

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
    tokened = nltk.word_tokenize(text.decode('utf8', 'ignore').replace('<br />',' '))
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

pos_count = 0
processed = 0
for review in pos_text:
    pos_score,neg_score = evaluate_sentiment(review)
    if pos_score > neg_score:
        pos_count += 1
    processed += 1
    sys.stdout.write('percent of %i positive reviews (total %i) classified as positive: %.2f%% \r' % (processed, len(pos_text), float(pos_count)/processed*100))
    sys.stdout.flush()

print 'percent of %i positive reviews (total %i) classified as positive: %.2f%% \r' % (processed, len(pos_text), float(pos_count)/processed*100)

neg_count = 0
processed = 0
for review in neg_text:
    pos_score,neg_score = evaluate_sentiment(review)
    if pos_score < neg_score:
        neg_count += 1
    processed += 1
    sys.stdout.write('percent of %i negative reviews (total %i) classified as negative: %.2f%% \r' % (processed, len(neg_text), float(neg_count)/processed*100))
    sys.stdout.flush()

print 'percent of %i negative reviews (total %i) classified as negative: %.2f%% \r' % (processed, len(neg_text), float(neg_count)/processed*100)