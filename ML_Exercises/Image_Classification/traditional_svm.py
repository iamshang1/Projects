import numpy as np
import mahotas as mh
from mahotas.features import surf
import glob
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split

print "loading images"
images = glob.glob('SimpleImageDataset/*.jpg')
features = []
labels = []
alldescriptors = []

def colors(image):
    image = image // 64
    r,g,b = image.transpose((2,0,1))
    pixels = 1 * r + 4 * b + 16 * g
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    hist = np.log1p(hist)
    return hist

for i in range(len(images)):
    print "processing image %i of %i" % (i+1, len(images)) 
    labels.append(images[i][:-len('00.jpg')])
    im = mh.imread(images[i])
    imgrey = mh.colors.rgb2gray(im, dtype=np.uint8)
    features.append(np.concatenate([mh.features.haralick(im).ravel(), mh.features.lbp(imgrey, 30, 10).ravel(), colors(im)]))
    surfim = mh.imread(images[i], as_grey=True)
    surfim = surfim.astype(np.uint8)
    alldescriptors.append(surf.dense(surfim, spacing=16))

concatenated = np.concatenate(alldescriptors)
print "fitting k mean clusters for surf descriptors"
km = KMeans(15)
km.fit(concatenated)
print "creating surf features"
sfeatures = []
for d in alldescriptors:
    c = km.predict(d)
    sfeatures.append(np.array([np.sum(c == ci) for ci in range(15)]))

features = np.array(features) 
sfeatures = np.array(sfeatures, dtype=float)
features = np.concatenate((features, sfeatures), axis=1)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42, stratify=labels)
clf = Pipeline([('scaler', StandardScaler()),('classifier', OneVsRestClassifier(SVC()))])

print "building model"
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print 'Accuracy of model: %.2f%%' % (score*100.)