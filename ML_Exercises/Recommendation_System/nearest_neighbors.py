import numpy as np
import pandas as pd
from sklearn import cross_validation
from scipy import sparse
from scipy.spatial import distance

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

#mean error using mean movie rating across all users
print "predicting rating using mean movie rating across all users"
#create dictionary of average rating for each movie
unique_movies = X_test.mid.unique()
average_ratings = {}
for movie in unique_movies:
    rating = X_train[X_train.mid == movie]['rating'].mean()
    average_ratings[movie] = rating
#get average rating for each movie in test set
y_pred = np.array([average_ratings[movie] for movie in X_test.mid])
#calculate error
error = np.sqrt(np.mean((y_test - y_pred)**2))
print "root mean square rating error using mean movie rating across all users: %.2f\n" % error

#mean error using collaborative filtering
print "predicting rating using ratings using collaborative filtering methods"
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

#plot data (eda purposes only)
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.figure(figsize=(15,15))
ax = plt.gca()
im = ax.imshow(user_rows, interpolation='nearest')
plt.title('matrix of movie ratings')
plt.xlabel('movie ids')
plt.ylabel('user ids')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
plt.savefig('sparse.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

#find nearest users and movies
print 'calculating nearest users'
user_dists = distance.pdist(user_rows, 'correlation')
user_dists = distance.squareform(user_dists)
closest_users = user_dists.argsort(axis=1)

print 'calculating nearest movies\n'
movie_dists = distance.pdist(movie_rows, 'correlation')
movie_dists = distance.squareform(movie_dists)
closest_movies = movie_dists.argsort(axis=1)

#predict movie ratings for test set entries
print "predicting movie ratings using ratings from most similar users"
y_pred = []
for index, row in X_test.iterrows():
    nearest_users = closest_users[row[0], 1:]
    ratings = user_rows[nearest_users, row[1]]
    no_nan = ratings[np.nonzero(ratings)]
    num_revs = len(no_nan)
    if num_revs == 0:
        mean_user_rating = average_ratings[row[1]]
    else:
        closest_ratings = no_nan[:num_revs/2+1]
        mean_user_rating = np.mean(closest_ratings)    
    y_pred.append(mean_user_rating)
#calculate error
y_pred = np.array(y_pred)
error = np.sqrt(np.mean((y_test - y_pred)**2))
print "root mean square rating error using ratings from most similar users: %.2f\n" % error

#predict movie ratings for test set entries
print "predicting movie ratings using user's ratings on most similar movies"
y_pred = []
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
    y_pred.append(mean_movie_rating)
#calculate error
y_pred = np.array(y_pred)
error = np.sqrt(np.mean((y_test - y_pred)**2))
print "root mean square rating error using user's ratings on most similar movies: %.2f\n" % error