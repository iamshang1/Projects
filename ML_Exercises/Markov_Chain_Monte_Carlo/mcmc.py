import re
import string
import numpy as np
import random
import copy
import math
import decimal

def remove_nonchars(corpus):
    corpus = corpus.replace("\n"," ")
    pattern = re.compile('[^a-zA-Z ]')
    corpus = pattern.sub('', corpus)
    corpus = re.sub(' +',' ',corpus)
    return corpus.lower()

def transition_matrix(corpus):
    chars = (list(string.ascii_lowercase) + [' '])
    transition_matrix = np.ones((len(chars),len(chars)))*0.01
    for i in range(len(corpus)-1):
        row = chars.index(corpus[i])
        col = chars.index(corpus[i+1])
        transition_matrix[row,col] += 1
    transition_matrix /= np.sum(transition_matrix,axis=0)
    return transition_matrix

def likelihood(corpus,transition_matrix):
    chars = (list(string.ascii_lowercase) + [' '])
    lik = decimal.Decimal(1)
    for i in range(len(corpus)-1):
        row = chars.index(corpus[i])
        col = chars.index(corpus[i+1])
        lik *= decimal.Decimal(transition_matrix[row,col])
    return lik
    
def random_mapping():
    chars = list(string.ascii_lowercase)
    scrambled = list(string.ascii_lowercase)
    random.shuffle(scrambled)
    map = dict(zip(chars,scrambled))
    return map

def scramble(corpus):
    map = random_mapping()
    scrambled = ""
    for i in range(len(corpus)):
        if corpus[i]==" ":
            scrambled += " "
        else:
            key = corpus[i]
            scrambled += map[key]
    return scrambled
    
def unscramble(corpus,map):
    unscrambled = ""
    for i in range(len(corpus)):
        if corpus[i]==" ":
            unscrambled += " "
        else:
            key = corpus[i]
            unscrambled += map[key]
    return unscrambled
    
def mcmc(corpus,map,transition_matrix,prev_prob):
    a = random.choice(map.keys())
    b = random.choice(map.keys())
    newmap = copy.copy(map)
    newmap[a] = map[b]
    newmap[b] = map[a]
    unscrambled = unscramble(corpus,newmap)
    probability = likelihood(unscrambled,transition_matrix)
    if probability > prev_prob:
        return newmap,probability
    elif random.random() < probability/prev_prob:
        return newmap,probability
    else:
        return map,prev_prob

with open('hhgttg.txt','r') as f:
    text = f.read()
corpus = remove_nonchars(text)
train = corpus[:-500]
test = corpus[-500:]
transition_matrix = transition_matrix(train)
test = scramble(test)
map = random_mapping()
probability = 0

attempts = open('attempts.txt','w')
attempts.write((str(0)+":\n "+test+"\n\n"))
for i in range(10000):
    map,probability = mcmc(test,map,transition_matrix,probability)
    print "step %i adjusted probability:" % (i+1), probability
    if (i+1)%250==0:
        unscrambled = unscramble(test,map)
        attempts.write((str(i+1)+":\n "+unscrambled+"\n\n"))
        print unscrambled