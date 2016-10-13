# Movie Recommendations

This exercises uses collaborative filtering with nearest neighbors and matrix factorization to predict movie ratings.
The MovieLens dataset with 100,000 samples is used for this exercise.

The data is split into a 10,000 sample test set and a 90,000 sample training set. 
The following methods are used to predict a user's rating on a new movie:
 - Simply use the average movie rating across all users to predict the rating for the new movie
 - Find all other users who have rated the movie. Use correlation-based nearest neighbors to rank the other users by similarity 
   to the predicted user. Use the average movie rating across the top 50% most similar users to predict the rating for the new movie. 
 - Find all other movies the user has rated. Use correlation-based nearest neighbors to rank the other movies by similarity 
   to the predicted movie. Use the average movie rating across the top 10 most similar movies to predict the rating for the new movie. 
 - Use gradient descent matrix factorization on the user X movies matrix to create two smaller matrices representing latent user features
   and latent movie features (50 features each). Use these feature matrices to predict missing movie ratings.
 - Combine the best performing nearest neighbors method with matrix factorization using linear regression to predict missing movie ratings.

## Visualizating the Data

The following image displays a sparse matrix with the movie ratings of each user. Most of the matrix is empty (blue) because
most users have only watched a few movies.

![sparse](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Recommendation_System/sparse.png)

## Collaborative Filtering Results

Root mean squared error is used to measure the predictive power of each method.

RMSE using average movie rating across all users: 1.02

RMSE using average movie rating across nearest users: 1.01

RMSE using average movie rating across nearest movies: 0.95

RMSE using matrix factorization without bias term: 1.13

RMSE using matrix factorization using average movie/user rating as bias term: 0.98

RMSE using average movie rating across nearest movies combined with matrix factorization with bias term: 0.92

We see about a 10% increase in movie rating prediction accuracy by combining the best performing nearest neighbor method with matrix factoriation.