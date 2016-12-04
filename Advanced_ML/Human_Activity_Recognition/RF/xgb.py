import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
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

        #xgboost classifier
        gbm = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300)
        gbm.fit(X_train, y_train)
        prediction = gbm.predict(X_test)
        
        #calculate accuracy
        acc = float(np.sum(y_test==prediction))/y_test.shape[0]
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
    
    #xgboost classifier
    gbm = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300)
    gbm.fit(X_train, y_train)
    prediction = gbm.predict(X_test)
    
    #calculate accuracy
    acc = float(np.sum(y_test==prediction))/y_test.shape[0]
    print "between-subject accuracy: %.4f" % (acc*100)