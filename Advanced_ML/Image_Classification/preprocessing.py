import subprocess
import urllib
import numpy as np
from PIL import Image
from scipy.linalg import svd
import theano
import theano.tensor as T

'''
download train and test images from S3 to local drive
convert images to numpy arrays and save
apply ZCA whitening and save

outputs:
.npy files of images in shape (images,3,32,32)
'''

#Read Input
X_train = "../data/X_train.txt"
X_test = "../data/X_test.txt"
files = [X_train,X_test]

#download files
filenames = []
for file in files:
    with open(file,'r') as f:
        filenames += f.readlines()

for i in xrange(len(filenames)):
    file = filenames[i].replace('\n','').strip()
    print 'Getting file %i of %i: %s' % (i+1,len(filenames),file)    
    imgopen = urllib.URLopener()
    imgopen.retrieve('https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/%s.png' % file, "./data/images/%s.png" % file)

#convert images to numpy arrays
for k in range(2):
    with open(files[k],'r') as f:
        filenames = f.readlines()

    inputs = []
        
    input = np.empty((0,3,32,32))
    for i in xrange(len(filenames)):
        if i % 1000 == 0:
            inputs.append(input)
            input = np.empty((0,3,32,32))
        print 'Loading file %i of %i:' % (i+1,len(filenames))
        file = filenames[i].replace('\n','').strip()
        image = Image.open("./data/images/%s.png" % file)
        img = np.array(image,dtype='float64')/256
        img = img.transpose(2, 0, 1).reshape(1,3,32,32)
        input = np.concatenate((input,img),axis=0)
        image.close()
    inputs.append(input)

    final = np.empty((0,3,32,32))
    for i in xrange(len(inputs)):
        print 'Combining chunks %i of %i:' % (i+1,len(inputs))
        final = np.concatenate((final,inputs[i]),axis=0)
    
    if k == 0:
        np.save('X_train', final)
    else:
        np.save('X_test', final)

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
        
X_train = np.load('X_train.npy')
X_train_shape = X_train.shape
X_train_flattened = X_train.reshape(X_train_shape[0],np.prod(X_train_shape[1:]))

X_test = np.load('X_test.npy')
X_test_shape = X_test.shape
X_test_flattened = X_test.reshape(X_test_shape[0],np.prod(X_test_shape[1:]))

X = np.concatenate((X_train_flattened,X_test_flattened))

zca = ZCA()
output = zca.fit_transform(X,10**-5)
X_train_output = output[:X_train_shape[0]]
X_test_output = output[X_train_shape[0]:]

X_train_output = X_train_output.reshape((X_train_shape[0],X_train_shape[1],X_train_shape[2],X_train_shape[3]))
X_test_output = X_test_output.reshape((X_test_shape[0],X_test_shape[1],X_test_shape[2],X_test_shape[3]))

np.save('X_train_zca', X_train_output)
np.save('X_test_zca', X_test_output)