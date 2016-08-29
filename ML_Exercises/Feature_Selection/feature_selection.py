import time
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#create data with junk features
print "generating random dataset with junk features"
X,y = make_classification(n_samples=20000, n_features=30, n_informative=10, n_redundant=5, n_repeated=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
start = time.clock()
clf = SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) * 100.
end = time.clock()
print "test score of support vector machine trained on initial data %.2f%%" % score
print "time to train on initial dataset: %.4f seconds\n" % (end - start)

#use univariate feature selection to eliminate junk features
print 'eliminating junk features using univariate feature selection'
features = []
accuracy = []
for i in range(6,20):
    start = time.clock()
    KBest = SelectKBest(f_classif, k=i)
    KBest.fit(X_train, y_train)
    X_train_kbest = KBest.transform(X_train)
    X_test_kbest = KBest.transform(X_test)
    end = time.clock()
    features.append(i)
    print "reduced features from 30 to %i" % i
    print "time to apply univariate feature selection: %.4f seconds" % (end - start)
    start = time.clock()
    clf = SVC()
    clf.fit(X_train_kbest, y_train)
    score = clf.score(X_test_kbest, y_test) * 100.
    accuracy.append(score)
    end = time.clock()
    print "test score of support vector machine trained on data passed through ufs %.2f%%" % score
    print "time to train on cleaned dataset: %.4f seconds\n" % (end - start)  

plt.plot(features, accuracy)
plt.title("Features Used vs Training Accuracy")
plt.xlabel('Number of Features')
plt.ylabel('Training Accuracy')
plt.savefig('ufs.png')
plt.show()

#apply recursive feature elimination to eliminate junk features
print 'eliminating junk features using recursive feature elimination'
features = []
accuracy = []
for i in range(6,20):
    start = time.clock()
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    myRFE = RFE(clf, n_features_to_select=i)
    myRFE = myRFE.fit(X_train,y_train)
    X_rfe = X_train[:,myRFE.support_]
    end = time.clock()
    features.append(i)
    print "reduced features from 30 to %i" % i
    print "time to apply recursive feature elimination: %.4f seconds" % (end - start)
    start = time.clock()
    clf = SVC()
    clf.fit(X_rfe, y_train)
    score = clf.score(X_test[:,myRFE.support_], y_test) * 100.
    accuracy.append(score)
    end = time.clock()
    print "test score of support vector machine trained on data passed through rfe %.2f%%" % score
    print "time to train on cleaned dataset: %.4f seconds\n" % (end - start)

plt.plot(features, accuracy)
plt.title("Features Used vs Training Accuracy")
plt.xlabel('Number of Features')
plt.ylabel('Training Accuracy')
plt.savefig('rfe.png')
plt.show()
 
#apply l1 lasso regularization to eliminate junk features
print 'eliminating junk features using l1 lasso regularization'
regularization = [0.0008,0.001,0.002,0.004,0.006,0.008,0.01,0.015,0.02,0.025,0.03]
accuracy = []
for i in regularization:
    start = time.clock()
    clf = LogisticRegression(penalty='l1',C=i)
    clf.fit(X_train,y_train)
    model = SelectFromModel(clf, prefit=True)
    X_train_l1 = model.transform(X_train)
    X_test_l1 = model.transform(X_test)
    end = time.clock()
    print "reduced features from 30 to %i" % X_train_l1.shape[1]
    print "time to apply l1 lasso: %.4f seconds" % (end - start)
    start = time.clock()
    clf = SVC()
    clf.fit(X_train_l1, y_train)
    score = clf.score(X_test_l1, y_test) * 100.
    accuracy.append(score)
    end = time.clock()
    print "test score of support vector machine trained on data passed through l1 lasso %.2f%%" % score
    print "time to train on cleaned dataset: %.4f seconds\n" % (end - start)  

plt.plot(regularization, accuracy)
plt.title("Regularization Strength vs Training Accuracy")
plt.xlabel('Inverse Regularization Strength')
plt.ylabel('Training Accuracy')
plt.savefig('l1.png')
plt.show()
 
#apply tree-based feature selection to eliminate junk features
print 'eliminating junk features using tree-based feature selection'
threshold = [0.06,0.055,0.05,0.045,0.04,0.035,0.03,0.025,0.02,0.015,0.01]
accuracy = []
for i in threshold:
    start = time.clock()
    clf = ExtraTreesClassifier()
    clf.fit(X_train,y_train)
    model = SelectFromModel(clf, threshold=i, prefit=True)
    X_train_l1 = model.transform(X_train)
    X_test_l1 = model.transform(X_test)
    end = time.clock()
    print "reduced features from 30 to %i" % X_train_l1.shape[1]
    print "time to apply tree-based selection: %.4f seconds" % (end - start)
    start = time.clock()
    clf = SVC()
    clf.fit(X_train_l1, y_train)
    score = clf.score(X_test_l1, y_test) * 100.
    accuracy.append(score)
    end = time.clock()
    print "test score of support vector machine trained on data passed through tree-based selection %.2f%%" % score
    print "time to train on cleaned dataset: %.4f seconds\n" % (end - start)

plt.plot(threshold, accuracy)
plt.title("Feature Elimination Threshold vs Training Accuracy")
plt.xlabel('Feature Elimination Threshold')
plt.ylabel('Training Accuracy')
plt.gca().invert_xaxis()
plt.savefig('tree.png')
plt.show()
    
#apply principal component analysis to reduce dimensionality
print 'reducing dimensionality using principal component analysis'
features = []
accuracy = []
for i in range(6,20):
    start = time.clock()
    pca = PCA()
    pca.fit(X_train)
    pca.set_params(n_components=i)
    X_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    end = time.clock()
    features.append(i)
    print "reduced dimensions from 30 to %i" % X_pca.shape[1]
    print "time to apply principal component analysis: %.4f seconds" % (end - start)
    start = time.clock()
    clf = SVC()
    clf.fit(X_pca, y_train)
    score = clf.score(X_test_pca, y_test) * 100.
    accuracy.append(score)
    end = time.clock()
    print "test score of support vector machine trained on data passed through pca %.2f%%" % score
    print "time to train on cleaned dataset: %.4f seconds\n" % (end - start)
    
plt.plot(features, accuracy)
plt.title("Number of Components vs Training Accuracy")
plt.xlabel('Number of Components')
plt.ylabel('Training Accuracy')
plt.savefig('pca.png')
plt.show()