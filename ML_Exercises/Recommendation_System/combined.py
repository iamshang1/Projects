import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn import cross_validation
from scipy import sparse
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

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

#matrix factorization
print "calculating matrix factorization"
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

y_pred_mf = []
for i in range(X_train.shape[0]):
    y_pred_mf.append(np.dot(P[X_train.iloc[i,0],:],Q[X_train.iloc[i,1],:])+bu[X_train.iloc[i,0]]+bm[X_train.iloc[i,1]])
y_pred_mf = np.array(y_pred_mf)

#nearest neighbors
print "calculating nearest neighbors"
unique_movies = X_test.mid.unique()
average_ratings = {}
for movie in unique_movies:
    rating = X_train[X_train.mid == movie]['rating'].mean()
    average_ratings[movie] = rating

movie_dists = distance.pdist(movie_rows, 'correlation')
movie_dists = distance.squareform(movie_dists)
closest_movies = movie_dists.argsort(axis=1)

y_pred_nn = []
for index, row in X_train.iterrows():
    nearest_movies = closest_movies[row[1], 1:]
    ratings = movie_rows[nearest_movies, row[0]]
    no_nan = ratings[np.nonzero(ratings)]
    num_revs = len(no_nan)
    if num_revs == 0:
        mean_movie_rating = average_ratings[row[1]]
    else:
        closest_ratings = no_nan[:num_revs/10+2]
        mean_movie_rating = np.mean(closest_ratings)    
    y_pred_nn.append(mean_movie_rating)
y_pred_nn = np.array(y_pred_nn)

#combine nearest neighbors and matrix factorization
print "combining recommendation systems using logistic regression"
y_pred = np.vstack((y_pred_mf,y_pred_nn)).T
lr = LinearRegression()
lr.fit(y_pred, y_train)

#calculate error
print "calculating test error"
y_pred_mf = []
for i in range(X_test.shape[0]):
    y_pred_mf.append(np.dot(P[X_test.iloc[i,0],:],Q[X_test.iloc[i,1],:])+bu[X_test.iloc[i,0]]+bm[X_test.iloc[i,1]])
y_pred_mf = np.array(y_pred_mf)

y_pred_nn = []
for index, row in X_test.iterrows():
    nearest_movies = closest_movies[row[1], 1:]
    ratings = movie_rows[nearest_movies, row[0]]
    no_nan = ratings[np.nonzero(ratings)]
    num_revs = len(no_nan)
    if num_revs == 0:
        mean_movie_rating = average_ratings[row[1]]
    else:
        closest_ratings = no_nan[:num_revs/10+2]
        mean_movie_rating = np.mean(closest_ratings)    
    y_pred_nn.append(mean_movie_rating)
y_pred_nn = np.array(y_pred_nn)

y_pred = np.vstack((y_pred_mf,y_pred_nn)).T
y_pred = lr.predict(y_pred)
error = np.sqrt(np.mean((y_test - y_pred)**2))
print "root mean square rating error using matrix factorization and nearest neighbors: %.2f\n" % error