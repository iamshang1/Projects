import pandas as pd
import numpy as np
import glob
import sys
import os

# verify the required arguments are given
if (len(sys.argv) < 3):
    print 'Usage: python preprocessing.py <PATH_TO_DEMOGRAPHICS_CSV> <PATH_TO_ACCELEROMETER_DATA_FOLDER>'
    exit(1)

#load demographics data
demographics = pd.read_csv(sys.argv[1],index_col=0,usecols=[0,2,3,4,5,6,7])

#one hot encoding for body type
demographics['overweight'] = (demographics['Type'] == 'Overweight').astype(int)
demographics['obese'] = (demographics['Type'] == 'Obese').astype(int)
demographics = demographics.drop('Type', 1)

#normalize age, height, weight, bmi
demographics['age'] = (demographics['age'] - demographics['age'].mean()) / demographics['age'].std()
demographics['height'] = (demographics['height'] - demographics['height'].mean()) / demographics['height'].std()
demographics['weight'] = (demographics['weight'] - demographics['weight'].mean()) / demographics['weight'].std()
demographics['BMI'] = (demographics['BMI'] - demographics['BMI'].mean()) / demographics['BMI'].std()

#load accelerometer data
files = glob.glob(sys.argv[2]+'/*')
for file in files:
    data = pd.read_csv(file,header=None,index_col=False,usecols=[0,2,3,4,5,6],names=['rowID','activity','x','y','z','ID'])
    
    #split by activity
    activities = data.activity.unique()
    for activity in activities:
        mask = data.activity==activity
        
        #join with demographics data based on person ID
        chunk = data[mask].join(demographics,on='ID')
        
        #if no demographic data for person ID, skip activity chunk
        if np.sum(np.sum(pd.isnull(chunk))) > 0:
            continue
        
        #find any disconnects in rowids (same activity performed at different times)
        rowIDs = chunk.rowID.reset_index()
        shifted = rowIDs.shift(1)
        diff = rowIDs-shifted
        disconnects = diff!=1
        disconnects = disconnects[disconnects].dropna().index.tolist()
        disconnects.append(chunk.shape[0])
        
        #drop unneeded columns
        chunk = chunk.drop('activity',1)
        chunk = chunk.drop('rowID',1)
        UID = chunk.iloc[0,3]
        chunk = chunk.drop('ID',1)
        
        #create target directory
        if not os.path.exists('./data/Arrays/'):
            os.makedirs('./data/Arrays/')
        
        #save to numpy array
        chunk = np.array(chunk)
        for i in range(len(disconnects)-1):        
            subchunk = chunk[disconnects[i]:disconnects[i+1],:]
            print "saving UID %s, activity %s" % (UID,activity)
            np.save('./data/Arrays/%s_%s_%s' % (UID,activity,i),subchunk)