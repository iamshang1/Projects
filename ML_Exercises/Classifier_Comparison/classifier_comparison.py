import numpy as np
import pandas as pd
import zipfile
import gzip, cPickle
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import time

# Load the titanic dataset
titanic = pd.read_csv('titanic.csv', sep=',', header=0, usecols=(1,2,4,5,6,7))
for i in range(titanic.shape[0]):
    titanic.iloc[i,2] = 1 if titanic.iloc[i,2] == "male" else 0
titanic = titanic.dropna(0)
X = titanic.iloc[:,1:]
X = scale(X)
y = titanic.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#naive bayes on titanic
start = time.clock()
clf = GaussianNB()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "naive bayes accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train naive bayes: %.2f seconds\n" % (end - start)

#logistic regression on titanic
start = time.clock()
clf = LogisticRegression('l2', C=1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "logistic regression accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train logistic regression: %.2f seconds\n" % (end - start)

#support vector machine w/ linear kernel on titanic
start = time.clock()
clf = LinearSVC(C=0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "linear support vector machine accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train linear support vector machine: %.2f seconds\n" % (end - start)

#support vector machine w/ rbf kernel on titanic
start = time.clock()
clf = SVC(C=1, kernel='rbf', gamma = 0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "rbf support vector machine accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train rbf support vector machine: %.2f seconds\n" % (end - start)

#random forest on titanic
start = time.clock()
clf = RandomForestClassifier(100)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "random forest accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train random forest: %.2f seconds\n" % (end - start)

#adaboost on titanic
start = time.clock()
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "adaboost accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train adaboost: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ euclidean distance on titanic
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "euclidean k nearest neighbors accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train euclidean k nearest neighbors: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ cosine distance on titanic
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "cosine k nearest neighbors accuracy on titanic dataset: %.2f%%" % accuracy
print "time to train cosine k nearest neighbors: %.2f seconds\n" % (end - start)

'''
-----------------------------------------
'''
print "\n-----------------------------------------\n\n"

# Load the MAGIC dataset
zf = zipfile.ZipFile('magic.zip')
data = zf.open('magic.dat')
magic = pd.read_csv(data, sep=',', skiprows=15)
X = magic.iloc[:,:-1]
y = magic.iloc[:,-1]
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#naive bayes on MAGIC
start = time.clock()
clf = GaussianNB()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "naive bayes accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train naive bayes: %.2f seconds\n" % (end - start)

#logistic regression on MAGIC
start = time.clock()
clf = LogisticRegression('l2', C=1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "logistic regression accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train logistic regression: %.2f seconds\n" % (end - start)

#support vector machine w/ linear kernel on MAGIC
start = time.clock()
clf = LinearSVC(C=0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "linear support vector machine accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train linear support vector machine: %.2f seconds\n" % (end - start)

#support vector machine w/ rbf kernel on MAGIC
start = time.clock()
clf = SVC(C=1, kernel='rbf', gamma = 0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "rbf support vector machine accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train rbf support vector machine: %.2f seconds\n" % (end - start)

#random forest on MAGIC
start = time.clock()
clf = RandomForestClassifier(100)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "random forest accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train random forest: %.2f seconds\n" % (end - start)

#adaboost on MAGIC
start = time.clock()
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "adaboost accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train adaboost: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ euclidean distance on MAGIC
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "euclidean k nearest neighbors accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train euclidean k nearest neighbors: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ cosine distance on MAGIC
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "cosine k nearest neighbors accuracy on MAGIC dataset: %.2f%%" % accuracy
print "time to train cosine k nearest neighbors: %.2f seconds\n" % (end - start)

'''
-----------------------------------------
'''
print "\n-----------------------------------------\n\n"

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#naive bayes on digits
start = time.clock()
clf = OneVsRestClassifier(MultinomialNB())
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "naive bayes accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train naive bayes: %.2f seconds\n" % (end - start)

#logistic regression on digits
start = time.clock()
clf = LogisticRegression('l1', C=0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "logistic regression accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train logistic regression: %.2f seconds\n" % (end - start)

#support vector machine w/ linear kernel on digits
start = time.clock()
clf = LinearSVC(C=0.1)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "linear support vector machine accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train linear support vector machine: %.2f seconds\n" % (end - start)

#support vector machine w/ rbf kernel on digits
start = time.clock()
clf = OneVsRestClassifier(SVC(C=1, kernel='rbf', gamma=0.001))
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "rbf support vector machine accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train rbf support vector machine: %.2f seconds\n" % (end - start)

#random forest on digits
start = time.clock()
clf = RandomForestClassifier(100)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "random forest accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train random forest: %.2f seconds\n" % (end - start)

#adaboost on digits
start = time.clock()
clf = OneVsRestClassifier(AdaBoostClassifier())
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "adaboost accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train adaboost: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ euclidean distance on digits
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "euclidean k nearest neighbors accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train euclidean k nearest neighbors: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ cosine distance on digits
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "cosine k nearest neighbors accuracy on digits (small dataset): %.2f%%" % accuracy
print "time to train cosine k nearest neighbors: %.2f seconds\n" % (end - start)

'''
-----------------------------------------
'''
print "\n-----------------------------------------\n\n"

# Load the mnist dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train = np.array(train_set[0])
y_train = np.array(train_set[1])
X_test = np.array(test_set[0])
y_test = np.array(test_set[1])

#naive bayes on mnist
start = time.clock()
clf = OneVsRestClassifier(MultinomialNB())
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "naive bayes accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train naive bayes: %.2f seconds\n" % (end - start)

#logistic regression on mnist
start = time.clock()
clf = LogisticRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "logistic regression accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train logistic regression: %.2f seconds\n" % (end - start)

#support vector machine w/ linear kernel on mnist
start = time.clock()
clf = LinearSVC()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "linear support vector machine accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train linear support vector machine: %.2f seconds\n" % (end - start)

#support vector machine w/ rbf kernel on mnist
start = time.clock()
clf = OneVsRestClassifier(SVC(C=1, kernel='rbf', gamma=0.001))
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "rbf support vector machine accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train rbf support vector machine: %.2f seconds\n" % (end - start)

#random forest on mnist
start = time.clock()
clf = RandomForestClassifier(100)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "random forest accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train random forest: %.2f seconds\n" % (end - start)

#adaboost on mnist
start = time.clock()
clf = OneVsRestClassifier(AdaBoostClassifier())
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "adaboost accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train adaboost: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ euclidean distance on mnist
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "euclidean k nearest neighbors accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train k nearest neighbors: %.2f seconds\n" % (end - start)

#k nearest neighbors w/ cosine distance on mnist
start = time.clock()
clf = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test) * 100.0
end = time.clock()
print "cosine k nearest neighbors accuracy on mnist (large dataset): %.2f%%" % accuracy
print "time to train cosine k nearest neighbors: %.2f seconds\n" % (end - start)