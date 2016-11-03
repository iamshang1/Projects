'''
Saves numpy array of mean pixel values across all images (clips any pixel intensities above 3 std)
For training data, also saves the mask that indicates which pixels belong to ROI

Arguments:
Path to neurofinder folder, e.g. 'C:\Users\Username\Desktop\neurofinder.00.00'
Path to save output numpy arrays

Outputs:
X_<dataset> : Numpy array of clipped mean pixel values across all images
y_<dataset> : Numpy array of pixels belonging to ROI if regions json is provided
'''

import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy import array, zeros
from scipy.misc import imread
from glob import glob
import numpy as np
import os
import sys

# verify the required arguments are given
if (len(sys.argv) < 3):
    print 'Usage: python preprocessing.py <PATH_TO_NEUROFINDER_FOLDER> <PATH_TO_OUTPUT_FOLDER>'
    exit(1)
    
# load the images
files = sorted(glob(sys.argv[1] + '/images/*.tiff'))
imgs = array([imread(f) for f in files])
dims = imgs.shape[1:]

# name of current dataset
path = os.getcwd().split("\\")[-1]

# load the regions (training data only)
if os.path.isfile(sys.argv[1] + '/regions/regions.json'):

    with open(sys.argv[1] + '/regions/regions.json') as f:
        regions = json.load(f)

    #convert json to numpy array
    def tomask(coords):
        mask = zeros(dims)
        mask[zip(*coords)] = 1
        return mask

    masks = array([tomask(s['coordinates']) for s in regions])

    #get edge pixels of each region for plotting purposes
    def outline(mask):
        horz = np.append(np.diff(mask,axis=1),np.zeros((mask.shape[0],1,mask.shape[2])),1)
        horz[horz!=0]==1
        vert = np.append(np.diff(mask,axis=2),np.zeros((mask.shape[0],mask.shape[1],1)),2)
        vert[vert!=0]==1
        r_horz = np.append(np.diff(mask[:,::-1,:],axis=1),np.zeros((mask.shape[0],1,mask.shape[2])),1)[:,::-1,:]
        r_horz[r_horz!=0]==1
        r_vert = np.append(np.diff(mask[:,:,::-1],axis=2),np.zeros((mask.shape[0],mask.shape[1],1)),2)[:,:,::-1]
        r_vert[r_vert!=0]==1
        comb = horz+vert+r_horz+r_vert
        return comb

    outlines = outline(masks)
    
    #flatten regions of interest
    masks = masks.sum(axis=0)
    masks[masks!=0]==1
    
    #save numpy array
    np.save(sys.argv[2]+"/y_"+path,masks)

#transparent colormap
colors = [(1,0,0,i) for i in np.linspace(0,1,3)]
cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=10)

#generate mean intensity pixels over time
sum = np.sum(imgs,axis=0)

#clip pixels above 99 percentile intensity
mean = np.mean(sum)
std = np.std(sum)
print "min:",np.amin(sum)
print "max:",np.amax(sum)
print "mean:",mean
print "std:",std
sum[sum > (mean + 3*std)] = mean + 3*std 
sum = sum/mean

#save numpy array
np.save(sys.argv[2]+"/X_"+path,sum)

#show the outputs
if os.path.isfile(sys.argv[1] + '/regions/regions.json'):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(sum, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(sum, cmap='gray')
    plt.imshow(outlines.sum(axis=0), cmap=cmap, vmin=0, vmax=1)
    plt.show()
    
else:
    plt.figure()
    plt.imshow(sum, cmap='gray')
    plt.show()