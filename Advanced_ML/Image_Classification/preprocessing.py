import os
import glob
import numpy as np
from scipy.linalg import svd
import theano
import theano.tensor as T
import cPickle
import sys
import urllib2
import tarfile

'''
download cifar-10 dataset if not already downloaded
convert images to numpy arrays and save
apply ZCA whitening and save

outputs:
.npy files of images in shape (images,3,32,32)
'''

#download cifar-10 dataset if not found
if not os.path.isdir('./cifar-10-batches-py') and not os.path.isfile('cifar-10-python.tar.gz'):

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s\nTotal Bytes: %i" % (url, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        p = float(file_size_dl) / file_size
        status = r"{0}  [{1:.2%}]".format(file_size_dl, p)
        status = status + chr(8)*(len(status))
        sys.stdout.write(status)

    f.close()

#extract cifar-10 dataset if not already extracted
if not os.path.isdir('./cifar-10-batches-py'):
    print 'extracting cifar-10-python.tar.gz'
    tar = tarfile.open('cifar-10-python.tar.gz')
    tar.extractall()
    tar.close()
    
#function to unpickle cifar-10 images
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

train = glob.glob('./cifar-10-batches-py/data_*')
test = glob.glob('./cifar-10-batches-py/test_*')

#convert cifar-10 dataset to test and train numpy arrays
X_train = np.empty((0,3,32,32))
y_train = np.empty(0)
for file in train:
    dict = unpickle(file)
    X_train = np.concatenate((X_train,dict['data'].reshape(10000,3,32,32)),axis=0)
    y_train = np.concatenate((y_train,dict['labels']))
    
X_test = np.empty((0,3,32,32))
y_test = np.empty(0)
for file in test:
    dict = unpickle(file)
    X_test = np.concatenate((X_test,dict['data'].reshape(10000,3,32,32)),axis=0)
    y_test = np.concatenate((y_test,dict['labels']))
    
np.save('y_train', y_train)
np.save('y_test', y_test)
    
#zca whitening
#credit goes to https://github.com/sdanaipat/Theano-ZCA
print 'applying ZCA whitening'

class ZCA(object):
    def __init__(self):
        X_in = T.matrix('X_in')
        u = T.matrix('u')
        s = T.vector('s')
        eps = T.scalar('eps')

        X_ = X_in - T.mean(X_in, 0)
        sigma = T.dot(X_.T, X_) / X_.shape[0]
        self.sigma = theano.function([X_in], sigma, allow_input_downcast=True)

        Z = T.dot(T.dot(u, T.nlinalg.diag(1. / T.sqrt(s + eps))), u.T)
        X_zca = T.dot(X_, Z.T)
        self.compute_zca = theano.function([X_in, u, s, eps], X_zca, allow_input_downcast=True)

        self._u = None
        self._s = None

    def fit(self, X):
        cov = self.sigma(X)
        u, s, _ = svd(cov)
        self._u = u.astype(np.float32)
        self._s = s.astype(np.float32)
        del cov

    def transform(self, X, eps):
        return self.compute_zca(X, self._u, self._s, eps)

    def fit_transform(self, X, eps):
        self.fit(X)
        return self.transform(X, eps)
        
X_train_shape = X_train.shape
X_train_flattened = X_train.reshape(X_train_shape[0],np.prod(X_train_shape[1:]))

X_test_shape = X_test.shape
X_test_flattened = X_test.reshape(X_test_shape[0],np.prod(X_test_shape[1:]))

X = np.concatenate((X_train_flattened,X_test_flattened))

zca = ZCA()
output = zca.fit_transform(X,10**-5)
X_train_output = output[:X_train_shape[0]]
X_test_output = output[X_train_shape[0]:]

X_train_output = X_train_output.reshape((X_train_shape[0],X_train_shape[1],X_train_shape[2],X_train_shape[3]))
X_test_output = X_test_output.reshape((X_test_shape[0],X_test_shape[1],X_test_shape[2],X_test_shape[3]))

np.save('X_train', X_train_output)
np.save('X_test', X_test_output)