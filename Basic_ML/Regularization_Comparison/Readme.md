# Regularization Comparison

This exercise compares the effectiveness of different types of regularized linear regression on dense and sparse datasets.

The following datasets are used:
 - Dense dataset - Scikit Learn Boston Dataset (506 samples, 13 features)
 - Sparse dataset - Subsample from UCI Blog Feedback Dataset (5000 samples, 281 features)

Regularization prevents overfitting to data by penalizing models for attaching large coefficients to features.
The following regularization types are compared:
 - L1 Lasso (minimize sum of absolute value of coefficients)
 - L2 Ridge (minimize sum of squared coefficients)
 - L1 + L2 Elastic Net (combination of L1 + L2 regularization)
 
K-folds cross validation with 5 folds is used to test the effectiveness of each regularization method. Cross validation accuracy is
measured using root mean squared error and r-squared coefficient of determination.
 
### Dense Dataset Results

![dense_rmse](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Regularization_Comparison/dense_rmse.png)

![dense_r2](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Regularization_Comparison/dense_r2.png)

### Sparse Dataset Results

**Note:** Ordinary Least Squares Regression RMSE was around 3.9x10^14 and r2 was around -1.2x10^26. These have been omitted from 
the following charts.

![sparse_rmse](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Regularization_Comparison/sparse_rmse.png)

![sparse_r2](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Regularization_Comparison/sparse_r2.png)

### Conclusions

We see that L2 regression is more effective for dense datasets, while L1 regression is more effective for sparse datasets.