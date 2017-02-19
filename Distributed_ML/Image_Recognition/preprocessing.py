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
    for i in xrange(len(filenames)):
        print 'Loading file %i of %i:' % (i+1,len(filenames))
        file = filenames[i].replace('\n','').strip()
        image = Image.open("./data/images/%s.png" % file)
        img = np.array(image,dtype='float64')/256
        img = img.transpose(2, 0, 1)
        inputs.append(img)
        image.close()
    
    final = np.array(inputs)
    
    if k == 0:
        np.save('X_train', final)
    else:
        np.save('X_test', final)