import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import os

#check if within-subject data exists
if os.path.isfile('X.npy') and os.path.isfile('y.npy'):

    #load data
    X = np.load('X.npy')
    y = np.load('y.npy')

    #kfold cross validation
    splits = 10
    kf = KFold(n_splits=splits)
    score = []

    #classify across each fold
    i = 0
    for train_index, test_index in kf.split(X):
        i += 1
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #random forest classifier
        rf = RandomForestClassifier(100)
        rf.fit(X_train,y_train)
        acc = rf.score(X_test,y_test)
        print "kfold %i of %i accuracy: %.4f" % (i,splits,acc*100)
        score.append(acc)

    #average test accuracy
    print "average within-subject accuracy: %.4f" % (np.mean(score)*100)
    
#check if between-subject data exists
if os.path.isfile('X_train.npy') and os.path.isfile('y_train.npy') and os.path.isfile('X_test.npy') and os.path.isfile('y_test.npy'):
    
    #load data
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    #random forest classifier
    rf = RandomForestClassifier(100)
    rf.fit(X_train,y_train)
    acc = rf.score(X_test,y_test)
    print "between-subject accuracy: %.4f" % (acc*100)