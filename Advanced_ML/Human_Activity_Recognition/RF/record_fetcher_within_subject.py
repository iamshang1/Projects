import numpy as np
import glob
import sys
import matplotlib.pyplot as plt

class record_fetcher(object):
    '''
    creates feature arrays and labels from raw accelerometer/demographic data
    no test/train splitting is included
    
    methods:
      - fetch(batch_size,binary)
        calculates summary statistics from raw accelerometer/demographic data and creates
        input features and labels for random forest/xgboost classifiers
        parameters:
          - batch_size: integer
            number of frames to use for each set of summary statistics
            e.g. 1000 means calculate summary statistics over 10 (nonoverlapping) seconds across all data
          - binary: boolean (default True)
            use True to set labels for ambulatory/non-ambulatory
            use False to set labels for non-ambulatory/walking/running/upstairs/downstairs
        outputs:
          - numpy array representing summary statistics and demographic data over each time window
          - numpy array representing activity label over each time window
    '''
    
    def __init__(self):

        #categorize activity ids into ambulatory/non-ambulatory
        self.dic1 = {
        'ambulatory': [11,12,13,14,23,24,25,26,27,28,29,30,31,32,16,17,18,33,34],
        'nonambulatory': [19,20,21,22]
        }

        #categorize activity ids into non-ambulatory/walking/running/upstairs/downstairs
        self.dic2 = {
        'nonambulatory': [19,20,21,22],
        'walking': [11,12,13,14,23,24,25,26,27,28,29,30,31,32],
        'running': [16,17,18],
        'upstairs': [33],
        'downstairs': [34]
        }

        #get filenames for all activity arrays
        self.ambulatory = []
        for i in self.dic1['ambulatory']:
            self.ambulatory.extend(glob.glob('../data/arrays/*_%i_*' % i))

        self.nonambulatory = []
        for i in self.dic1['nonambulatory']:
            self.nonambulatory.extend(glob.glob('../data/arrays/*_%i_*' % i))
            
        self.walking = []
        for i in self.dic2['walking']:
            self.walking.extend(glob.glob('../data/arrays/*_%i_*' % i))
            
        self.running = []
        for i in self.dic2['running']:
            self.running.extend(glob.glob('../data/arrays/*_%i_*' % i))
            
        self.upstairs = []
        for i in self.dic2['upstairs']:
            self.upstairs.extend(glob.glob('../data/arrays/*_%i_*' % i))
            
        self.downstairs = []
        for i in self.dic2['downstairs']:
            self.downstairs.extend(glob.glob('../data/arrays/*_%i_*' % i))
        
    def fetch(self,batch_size,binary=True):
        '''
        calculates summary statistics from raw accelerometer/demographic data and creates
        input features and labels for random forest/xgboost classifiers
        parameters:
          - batch_size: integer
            number of frames to use for each set of summary statistics
            e.g. 1000 means calculate summary statistics over 10 (nonoverlapping) seconds across all data
          - binary: boolean
            use True to set labels for ambulatory/non-ambulatory
            use False to set labels for non-ambulatory/walking/running/upstairs/downstairs
        outputs:
          - numpy array representing summary statistics and demographic data over each time window
          - numpy array representing activity label over each time window
        '''
        
        X_list = []
        y_list = []
        
        #for ambulatory/non-ambulatory classification
        if binary:
        
            for a in self.ambulatory:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(1)
                
            for a in self.nonambulatory:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(0)
        
        #for non-ambulatory/walking/running/upstairs/downstairs classification
        else: 
            for a in self.nonambulatory:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(0)
                
            for a in self.walking:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(1)
                    
            for a in self.running:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(2)
                    
            for a in self.upstairs:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(3)
                    
            for a in self.downstairs:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
                
                #separate array into windows and calculate summary satistics per window
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    subarray = array[i*batch_size:(i+1)*batch_size,:]
                    features = self._create_features(subarray)
                    X_list.append(features)
                
                    #create label
                    y_list.append(4)
        
        #concatenate features
        X = np.concatenate(X_list,0)
        y = np.array(y_list)
        y = np.expand_dims(y, axis=1)
        
        #shuffle arrays
        comb = np.concatenate((X,y),1)
        np.random.shuffle(comb)
        X = comb[:,:-1]
        y = comb[:,-1]
        
        print 'feature vector shape:', X.shape
        
        return X, y
        
    def _create_features(self,array):
        '''
        calculate summary statistics over time window
        concatenate with normalized demographic data
        
        the following features are calculated for each axis (X,Y,Z), 
        magnitude (sqrt of X^2+Y^2+Z^2), first differential of each axis, 
        and first differential of magnitude:
         - mean, std, min, max
         - 10,25,50,75,90 percentiles
         - number of median crossings
         - correlation with other axis
        '''
        
        #create features
        mag = np.sqrt(array[:,0]**2+array[:,1]**2+array[:,2]**2)
        x_mean = np.mean(array[:,0])
        y_mean = np.mean(array[:,1])
        z_mean = np.mean(array[:,2])
        mag_mean = np.mean(mag)
        x_std = np.std(array[:,0])
        y_std = np.std(array[:,1])
        z_std = np.std(array[:,2])
        mag_std = np.std(mag)
        x_10per = np.percentile(array[:,0],10)
        x_25per = np.percentile(array[:,0],25)
        x_50per = np.percentile(array[:,0],50)
        x_75per = np.percentile(array[:,0],75)
        x_90per = np.percentile(array[:,0],90)
        x_med = np.median(array[:,0])
        x_medcross = np.sum(np.diff((array[:,0]==x_med).astype(int))==1)
        x_max = np.amax(array[:,0])
        x_min = np.amin(array[:,0])
        x_range = x_max - x_min
        x_iqrange = x_75per - x_25per
        y_10per = np.percentile(array[:,1],10)
        y_25per = np.percentile(array[:,1],25)
        y_50per = np.percentile(array[:,1],50)
        y_75per = np.percentile(array[:,1],75)
        y_90per = np.percentile(array[:,1],90)
        y_med = np.median(array[:,1])
        y_medcross = np.sum(np.diff((array[:,1]==y_med).astype(int))==1)
        y_max = np.amax(array[:,1])
        y_min = np.amin(array[:,1])
        y_range = y_max - y_min
        y_iqrange = y_75per - y_25per
        z_10per = np.percentile(array[:,2],10)
        z_25per = np.percentile(array[:,2],25)
        z_50per = np.percentile(array[:,2],50)
        z_75per = np.percentile(array[:,2],75)
        z_90per = np.percentile(array[:,2],90)
        z_med = np.median(array[:,2])
        z_medcross = np.sum(np.diff((array[:,2]==z_med).astype(int))==1)
        z_max = np.amax(array[:,2])
        z_min = np.amin(array[:,2])
        z_range = z_max - z_min
        z_iqrange = z_75per - z_25per
        mag_10per = np.percentile(mag,10)
        mag_25per = np.percentile(mag,25)
        mag_50per = np.percentile(mag,50)
        mag_75per = np.percentile(mag,75)
        mag_90per = np.percentile(mag,90)
        mag_med = np.median(mag)
        mag_medcross = np.sum(np.diff((mag==mag_med).astype(int))==1)
        mag_max = np.amax(mag)
        mag_min = np.amin(mag)
        mag_range = mag_max - mag_min
        mag_iqrange = mag_75per - mag_25per
        xy_corr = np.correlate(array[:,0],array[:,1])
        xz_corr = np.correlate(array[:,0],array[:,2])
        yz_corr = np.correlate(array[:,1],array[:,2])
        x_d1 = np.diff(array[:,0])
        y_d1 = np.diff(array[:,1])
        z_d1 = np.diff(array[:,2])
        mag_d1 = np.diff(mag)
        x_d1_mean = np.mean(x_d1)
        y_d1_mean = np.mean(y_d1)
        z_d1_mean = np.mean(z_d1)
        mag_mean_d1 = np.mean(mag_d1)
        x_d1_std = np.std(x_d1)
        y_d1_std = np.std(y_d1)
        z_d1_std = np.std(z_d1)
        mag_std_d1 = np.std(mag_d1)
        x_10per_d1 = np.percentile(x_d1,10)
        x_25per_d1 = np.percentile(x_d1,25)
        x_50per_d1 = np.percentile(x_d1,50)
        x_75per_d1 = np.percentile(x_d1,75)
        x_90per_d1 = np.percentile(x_d1,90)
        x_med_d1 = np.median(x_d1)
        x_medcross_d1 = np.sum(np.diff((x_d1==x_med_d1).astype(int))==1)
        x_max_d1 = np.amax(x_d1)
        x_min_d1 = np.amin(x_d1)
        x_range_d1 = x_max_d1 - x_min_d1
        x_iqrange_d1 = x_75per_d1 - x_25per_d1
        y_10per_d1 = np.percentile(y_d1,10)
        y_25per_d1 = np.percentile(y_d1,25)
        y_50per_d1 = np.percentile(y_d1,50)
        y_75per_d1 = np.percentile(y_d1,75)
        y_90per_d1 = np.percentile(y_d1,90)
        y_med_d1 = np.median(y_d1)
        y_medcross_d1 = np.sum(np.diff((y_d1==y_med_d1).astype(int))==1)
        y_max_d1 = np.amax(y_d1)
        y_min_d1 = np.amin(y_d1)
        y_range_d1 = y_max_d1 - y_min_d1
        y_iqrange_d1 = y_75per_d1 - y_25per_d1
        z_10per_d1 = np.percentile(z_d1,10)
        z_25per_d1 = np.percentile(z_d1,25)
        z_50per_d1 = np.percentile(z_d1,50)
        z_75per_d1 = np.percentile(z_d1,75)
        z_90per_d1 = np.percentile(z_d1,90)
        z_med_d1 = np.median(z_d1)
        z_medcross_d1 = np.sum(np.diff((z_d1==z_med_d1).astype(int))==1)
        z_max_d1 = np.amax(z_d1)
        z_min_d1 = np.amin(z_d1)
        z_range_d1 = z_max_d1 - z_min_d1
        z_iqrange_d1 = z_75per_d1 - z_25per_d1
        mag_10per_d1 = np.percentile(mag_d1,10)
        mag_25per_d1 = np.percentile(mag_d1,25)
        mag_50per_d1 = np.percentile(mag_d1,50)
        mag_75per_d1 = np.percentile(mag_d1,75)
        mag_90per_d1 = np.percentile(mag_d1,90)
        mag_med_d1 = np.median(mag_d1)
        mag_medcross_d1 = np.sum(np.diff((mag_d1==mag_med_d1).astype(int))==1)
        mag_max_d1 = np.amax(mag_d1)
        mag_min_d1 = np.amin(mag_d1)
        mag_range_d1 = mag_max_d1 - mag_min_d1
        mag_iqrange_d1 = mag_75per_d1 - mag_25per_d1
        xy_corr_d1 = np.correlate(x_d1,y_d1)
        xz_corr_d1 = np.correlate(x_d1,z_d1)
        yz_corr_d1 = np.correlate(y_d1,z_d1)
        
        #concatenate all features
        features = np.array([x_mean,x_mean,z_mean,x_std,y_std,z_std,xy_corr,xz_corr,yz_corr,\
            x_10per,x_25per,x_50per,x_75per,x_90per,x_max,x_min,x_medcross,x_range,x_iqrange,\
            y_10per,y_25per,y_50per,y_75per,y_90per,y_max,y_min,y_medcross,y_range,y_iqrange,\
            z_10per,z_25per,z_50per,z_75per,z_90per,z_max,z_min,z_medcross,z_range,z_iqrange,\
            mag_mean,mag_std,mag_10per,mag_25per,mag_50per,mag_75per,mag_90per,mag_max,mag_min,mag_medcross,mag_range,mag_iqrange,\
            x_d1_mean,y_d1_mean,z_d1_mean,x_d1_std,y_d1_std,z_d1_std,xy_corr_d1,xz_corr_d1,yz_corr_d1,\
            x_10per_d1,x_25per_d1,x_50per_d1,x_75per_d1,x_90per_d1,x_max_d1,x_min_d1,x_medcross_d1,x_range_d1,x_iqrange_d1,\
            y_10per_d1,y_25per_d1,y_50per_d1,y_75per_d1,y_90per_d1,y_max_d1,y_min_d1,y_medcross_d1,y_range_d1,y_iqrange_d1,\
            z_10per_d1,z_25per_d1,z_50per_d1,z_75per_d1,z_90per_d1,z_max_d1,z_min_d1,z_medcross_d1,z_range_d1,z_iqrange_d1,\
            mag_mean_d1,mag_std_d1,mag_10per_d1,mag_25per_d1,mag_50per_d1,mag_75per_d1,mag_90per_d1,mag_max_d1,mag_min_d1,mag_medcross_d1,mag_range_d1,mag_iqrange_d1])
        features = np.concatenate((features,array[0,3:]))
        
        return np.expand_dims(features, axis=0)
        
if __name__ == "__main__":

    # verify the required arguments are given
    if (len(sys.argv) < 2):
        print 'Usage: python record_fetcher_within_subject.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)
    
    if sys.argv[1] == '1':
        binary = True
    elif sys.argv[1] == '0':
        binary = False
    else:
        print 'Usage: python record_fetcher_within_subject.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)
        
    rf = record_fetcher()
    X,y = rf.fetch(1000,binary=binary)
    np.save('X',X)
    np.save('y',y)