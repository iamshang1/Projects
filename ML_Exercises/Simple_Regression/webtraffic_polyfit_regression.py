import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

#load data
data = np.genfromtxt('web_traffic.tsv', delimiter = '\t')

x = data[:,0].reshape(-1,1)
y = data[:,1]

x = x[~np.isnan(y)]
y = y[~np.isnan(y)]
print len(x)

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")

#plot polynomial regression on data
for degree in [1,2,3,4,5,10]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    extrapolate = np.arange(880).reshape(-1,1)
    y_plot = model.predict(extrapolate)
    plt.plot(extrapolate, y_plot, label="degree %d" % degree, linewidth=2)

plt.legend(loc='upper left')
plt.savefig('extrapolation.png')
plt.show()

# plot adjusted r2 vs polynomial regression on full data
adjusted_r2 = []
for degree in range(1,10):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, y)
    r2 = model.score(x,y)
    adjusted_r2.append(r2 - (1-r2)*(degree+1)/(len(x)-degree))

plt.plot(range(1,10), adjusted_r2)
plt.title("Training Adjusted R2 vs Polynomial Degree")
plt.xlabel('poly degree')
plt.ylabel('Training Adjusted R2')
plt.savefig('adjustedr2.png')
plt.show()

# plot k-folds cross validation error vs poly degree
cv_error=[]
kf = KFold(len(x), n_folds=15)
for degree in range(1,10):
    k_error=[]
    for train, test in kf:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x[train], y[train])
        k_error.append(model.score(x[test],y[test]))
    cv_error.append(np.mean(k_error))
  
plt.plot(range(1,10), cv_error)
plt.title("Cross Validation R2 vs Polynomial Degree")
plt.xlabel('poly degree')
plt.ylabel('Cross Validation R2')
plt.savefig('crossvalidation.png')
plt.show()