import subprocess
import urllib
import numpy as np
from PIL import Image

'''
download train and test images from S3 to local drive
convert images to numpy arrays and save

outputs:
.npy files of images in shape (images,3,32,32)
'''

#download files
files = ["./data/X_train.txt","./data/X_test.txt"]
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