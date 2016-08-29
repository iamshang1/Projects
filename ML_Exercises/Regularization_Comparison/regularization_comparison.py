import numpy as np
import zipfile
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt

#load dense dataset
boston = load_boston()
X = boston.data
y = boston.target

#perform ordinary least squares regression on full data
lr = LinearRegression()
lr.fit(X, y)
lr_pred = lr.predict(X)
lr_mse = mean_squared_error(y, lr_pred)
lr_cod = r2_score(y, lr_pred)
lr_rmse = np.sqrt(lr_mse)

#perform lasso regression on full data
la = Lasso(alpha=0.5)
la.fit(X, y)
la_pred = la.predict(X)
la_mse = mean_squared_error(y, la_pred)
la_cod = r2_score(y, la_pred)
la_rmse = np.sqrt(la_mse)

#perform ridge regression on full data
ri = Ridge(alpha=100)
ri.fit(X, y)
ri_pred = ri.predict(X)
ri_mse = mean_squared_error(y, ri_pred)
ri_cod = r2_score(y, ri_pred)
ri_rmse = np.sqrt(ri_mse)

#perform elastic net regression on full data
en = ElasticNet(alpha=0.5)
en.fit(X, y)
en_pred = en.predict(X)
en_mse = mean_squared_error(y, en_pred)
en_cod = r2_score(y, en_pred)
en_rmse = np.sqrt(en_mse)

print "ordinary least squares linear regression root mean square error on training set: %.3f" % lr_rmse
print "ordinary least squares linear regression coefficient of determination on training set: %.3f\n" % lr_cod
print "lasso linear regression root mean square error on training set: %.3f" % la_rmse
print "lasso linear regression coefficient of determination on training set: %.3f\n" % la_cod
print "ridge linear regression root mean square error on training set: %.3f" % ri_rmse
print "ridge linear regression coefficient of determination on training set: %.3f\n" % ri_cod
print "elastic net linear regression root mean square error on training set: %.3f" % en_rmse
print "elastic net linear regression coefficient of determination on training set: %.3f\n" % en_cod

#perform cross validated regression
lr_rmse_means = []
lr_cod_means = []
la_rmse_means = []
la_cod_means = []
ri_rmse_means = []
ri_cod_means = []
en_rmse_means = []
en_cod_means = []

lambdas = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for i in lambdas:
    kf = KFold(len(X), n_folds=5)
    lr_rmse = []
    lr_cod = []
    la_rmse = []
    la_cod = []
    ri_rmse = []
    ri_cod = []
    en_rmse = []
    en_cod = []
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        #least squares linear regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_cod.append(r2_score(y_test, lr_pred))
        lr_rmse.append(np.sqrt(lr_mse))
        
        #lasso linear regression
        la = Lasso(alpha=i)
        la.fit(X_train, y_train)
        la_pred = la.predict(X_test)
        la_mse = mean_squared_error(y_test, la_pred)
        la_cod.append(r2_score(y_test, la_pred))
        la_rmse.append(np.sqrt(la_mse))
        
        #ridge linear regression
        ri = Ridge(alpha=i*500)
        ri.fit(X_train, y_train)
        ri_pred = ri.predict(X_test)
        ri_mse = mean_squared_error(y_test, ri_pred)
        ri_cod.append(r2_score(y_test, ri_pred))
        ri_rmse.append(np.sqrt(ri_mse))
        
        #elastic net linear regression
        en = ElasticNet(alpha=i, l1_ratio=0.1)
        en.fit(X_train, y_train)
        en_pred = en.predict(X_test)
        en_mse = mean_squared_error(y_test, en_pred)
        en_cod.append(r2_score(y_test, en_pred))
        en_rmse.append(np.sqrt(en_mse))
        
    print "ordinary least squares linear regression root mean square error on cross validation set: %.3f" % np.mean(lr_rmse)
    print "ordinary least squares linear regression coefficient of determination on cross validation set: %.3f\n" % np.mean(lr_cod)
    print "lamba %.2f - lasso linear regression root mean square error on cross validation set: %.3f" % (i, np.mean(la_rmse))
    print "lamba %.2f - lasso linear regression coefficient of determination on cross validation set: %.3f\n" % (i, np.mean(la_cod))
    print "lamba %.2f - ridge linear regression root mean square error on cross validation set: %.3f" % (i*500, np.mean(ri_rmse))
    print "lamba %.2f - ridge linear regression coefficient of determination on cross validation set: %.3f\n" % (i*500, np.mean(ri_cod))
    print "lamba %.2f - elastic net linear regression root mean square error on cross validation set: %.3f" % (i, np.mean(en_rmse))
    print "lamba %.2f - elastic net linear regression coefficient of determination on cross validation set: %.3f\n" % (i, np.mean(en_cod))
        
    lr_rmse_means.append(np.mean(lr_rmse))
    lr_cod_means.append(np.mean(lr_cod))
    la_rmse_means.append(np.mean(la_rmse))
    la_cod_means.append(np.mean(la_cod))
    ri_rmse_means.append(np.mean(ri_rmse))
    ri_cod_means.append(np.mean(ri_cod))
    en_rmse_means.append(np.mean(en_rmse))
    en_cod_means.append(np.mean(en_cod))
    
lr, = plt.plot(lambdas, lr_rmse_means)
la, = plt.plot(lambdas, la_rmse_means)
ri, = plt.plot(lambdas, ri_rmse_means)
en, = plt.plot(lambdas, en_rmse_means)
plt.title("Cross Validation RMSE for Different Lamdas")
plt.xlabel('Lambda')
plt.ylabel('Cross Validation RMSE')
plt.legend([lr, la, ri, en], ['Ordinary Least Squares', 'Lasso', 'Ridge (Lambda*500)', 'Elastic Net'])
plt.savefig('dense_rmse.png')
plt.show()

lr, = plt.plot(lambdas, lr_cod_means)
la, = plt.plot(lambdas, la_cod_means)
ri, = plt.plot(lambdas, ri_cod_means)
en, = plt.plot(lambdas, en_cod_means)
plt.title("Cross Validation R2 for Different Lamdas")
plt.xlabel('Lambda')
plt.ylabel('Cross Validation R2')
plt.legend([lr, la, ri, en], ['Ordinary Least Squares', 'Lasso', 'Ridge (Lambda*500)', 'Elastic Net'], loc='lower right')
plt.savefig('dense_r2.png')
plt.show()

'''
-----------------------------------------
'''
print "\n-----------------------------------------\n\n"

#load sparse dataset
zf = zipfile.ZipFile('blog.zip')
data = zf.open('blog.csv')
blog = np.genfromtxt (data, delimiter=",")
X = blog[:,:-1]
y = blog[:,-1]
X = scale(X)

#perform ordinary least squares regression on full data
lr = LinearRegression()
lr.fit(X, y)
lr_pred = lr.predict(X)
lr_mse = mean_squared_error(y, lr_pred)
lr_cod = r2_score(y, lr_pred)
lr_rmse = np.sqrt(lr_mse)

#perform lasso regression on full data
la = Lasso(alpha=0.5)
la.fit(X, y)
la_pred = la.predict(X)
la_mse = mean_squared_error(y, la_pred)
la_cod = r2_score(y, la_pred)
la_rmse = np.sqrt(la_mse)

#perform ridge regression on full data
ri = Ridge(alpha=100)
ri.fit(X, y)
ri_pred = ri.predict(X)
ri_mse = mean_squared_error(y, ri_pred)
ri_cod = r2_score(y, ri_pred)
ri_rmse = np.sqrt(ri_mse)

#perform elastic net regression on full data
en = ElasticNet(alpha=0.5)
en.fit(X, y)
en_pred = en.predict(X)
en_mse = mean_squared_error(y, en_pred)
en_cod = r2_score(y, en_pred)
en_rmse = np.sqrt(en_mse)

print "ordinary least squares linear regression root mean square error on training set: %.3f" % lr_rmse
print "ordinary least squares linear regression coefficient of determination on training set: %.3f\n" % lr_cod
print "lasso linear regression root mean square error on training set: %.3f" % la_rmse
print "lasso linear regression coefficient of determination on training set: %.3f\n" % la_cod
print "ridge linear regression root mean square error on training set: %.3f" % ri_rmse
print "ridge linear regression coefficient of determination on training set: %.3f\n" % ri_cod
print "elastic net linear regression root mean square error on training set: %.3f" % en_rmse
print "elastic net linear regression coefficient of determination on training set: %.3f\n" % en_cod

#perform cross validated regression
lr_rmse_means = []
lr_cod_means = []
la_rmse_means = []
la_cod_means = []
ri_rmse_means = []
ri_cod_means = []
en_rmse_means = []
en_cod_means = []

lambdas = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]

for i in lambdas:
    kf = KFold(len(X), n_folds=5)
    lr_rmse = []
    lr_cod = []
    la_rmse = []
    la_cod = []
    ri_rmse = []
    ri_cod = []
    en_rmse = []
    en_cod = []
    for train, test in kf:
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        
        #ordinary least squares
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_cod = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(lr_mse)
        
        #lasso linear regression
        la = Lasso(alpha=i)
        la.fit(X_train, y_train)
        la_pred = la.predict(X_test)
        la_mse = mean_squared_error(y_test, la_pred)
        la_cod.append(r2_score(y_test, la_pred))
        la_rmse.append(np.sqrt(la_mse))
        
        #ridge linear regression
        ri = Ridge(alpha=i*2000)
        ri.fit(X_train, y_train)
        ri_pred = ri.predict(X_test)
        ri_mse = mean_squared_error(y_test, ri_pred)
        ri_cod.append(r2_score(y_test, ri_pred))
        ri_rmse.append(np.sqrt(ri_mse))
        
        #elastic net linear regression
        en = ElasticNet(alpha=i, l1_ratio=0.1)
        en.fit(X_train, y_train)
        en_pred = en.predict(X_test)
        en_mse = mean_squared_error(y_test, en_pred)
        en_cod.append(r2_score(y_test, en_pred))
        en_rmse.append(np.sqrt(en_mse))
    
    print "ordinary least squares linear regression root mean square error on training set: %.3f" % lr_rmse
    print "ordinary least squares linear regression coefficient of determination on training set: %.3f\n" % lr_cod    
    print "lamba %.2f - lasso linear regression root mean square error on cross validation set: %.3f" % (i, np.mean(la_rmse))
    print "lamba %.2f - lasso linear regression coefficient of determination on cross validation set: %.3f\n" % (i, np.mean(la_cod))
    print "lamba %.2f - ridge linear regression root mean square error on cross validation set: %.3f" % (i*2000, np.mean(ri_rmse))
    print "lamba %.2f - ridge linear regression coefficient of determination on cross validation set: %.3f\n" % (i*2000, np.mean(ri_cod))
    print "lamba %.2f - elastic net linear regression root mean square error on cross validation set: %.3f" % (i, np.mean(en_rmse))
    print "lamba %.2f - elastic net linear regression coefficient of determination on cross validation set: %.3f\n" % (i, np.mean(en_cod))

    lr_rmse_means.append(np.mean(lr_rmse))
    lr_cod_means.append(np.mean(lr_cod))    
    la_rmse_means.append(np.mean(la_rmse))
    la_cod_means.append(np.mean(la_cod))
    ri_rmse_means.append(np.mean(ri_rmse))
    ri_cod_means.append(np.mean(ri_cod))
    en_rmse_means.append(np.mean(en_rmse))
    en_cod_means.append(np.mean(en_cod))

la, = plt.plot(lambdas, la_rmse_means)
ri, = plt.plot(lambdas, ri_rmse_means)
en, = plt.plot(lambdas, en_rmse_means)
plt.title("Cross Validation RMSE for Different Lamdas")
plt.xlabel('Lambda')
plt.ylabel('Cross Validation RMSE')
plt.legend([la, ri, en], ['Lasso', 'Ridge (Lambda*2000)', 'Elastic Net'])
plt.savefig('sparse_rmse.png')
plt.show()

la, = plt.plot(lambdas, la_cod_means)
ri, = plt.plot(lambdas, ri_cod_means)
en, = plt.plot(lambdas, en_cod_means)
plt.title("Cross Validation R2 for Different Lamdas")
plt.xlabel('Lambda')
plt.ylabel('Cross Validation R2')
plt.legend([la, ri, en], ['Lasso', 'Ridge (Lambda*2000)', 'Elastic Net'], loc='lower right')
plt.savefig('sparse_r2.png')
plt.show()