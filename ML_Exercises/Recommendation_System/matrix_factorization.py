import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn import cross_validation
from scipy import sparse

#load data
print "loading data"
ratings = pd.read_csv('u.data', sep = "\t", names = ['uid','mid','rating','timestamp'], engine='python')
ratings.iloc[0,0] = 1
ratings['uid'] = pd.to_numeric(ratings['uid'])

#join datasets
print "creating test and train datasets\n"
data = ratings[['uid','mid','rating']]

#split into train and test
X = data
y = data['rating']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

#create sparse matrix of user ids and movie ids
matrix = sparse.coo_matrix((X_train['rating'],(X_train['uid'],X_train['mid'])))
user_rows = matrix.toarray().astype('float')
#make sure training matrix has 944 rows and 1683 cols in case test set calls idx not included in matrix
if user_rows.shape[1] < 1683:
    diff = 1683 - user_rows.shape[1]
    user_rows = np.concatenate((user_rows,np.zeros((user_rows.shape[0], diff))),axis=1)
if user_rows.shape[0] < 944:
    diff = 944 - user_rows.shape[0]
    user_rows = np.concatenate((user_rows,np.zeros((diff,user_rows.shape[1]))),axis=0)
movie_rows = user_rows.T

def matrix_factorization(R, P, Q, K, bu, bm, steps=1000, alpha=2e-7, beta=0.05):
    mask = R == 0
    R_ma = ma.masked_array(R,mask=mask)
    for step in xrange(steps):
        R_hat = np.dot(P,Q.T)
        R_hat_ma = ma.masked_array(R_hat,mask=mask)
        errors = (R_hat_ma + bu.reshape(len(bu),1) + bm) - R_ma
        p_gradients = np.dot(errors,Q)
        q_gradients = np.dot(P.T,errors).T
        P -= alpha*p_gradients + beta*P
        Q -= alpha*q_gradients + beta*Q
        print 'step %i error: %f' % (step,errors.mean())
        if errors.mean() < 0.01:
            break
    return P, Q

K = 50
P = np.random.rand(user_rows.shape[0],K)
Q = np.random.rand(movie_rows.shape[0],K)

mask = user_rows == 0
R_ma = ma.masked_array(user_rows,mask=mask)
bu = np.mean(R_ma,axis=1)/2
bm = np.mean(R_ma,axis=0)/2
bu = ma.filled(bu, fill_value=1.25)
bm = ma.filled(bm, fill_value=1.25)

P,Q = matrix_factorization(user_rows,P,Q,K,bu,bm)

#calculate error
y_pred = []

for i in range(X_test.shape[0]):
    y_pred.append(np.dot(P[X_test.iloc[i,0],:],Q[X_test.iloc[i,1],:])+bu[X_test.iloc[i,0]]+bm[X_test.iloc[i,1]])
y_pred = np.array(y_pred)
error = np.sqrt(np.mean((y_test - y_pred)**2))
print "root mean square rating error using matrix factorization: %.2f\n" % error