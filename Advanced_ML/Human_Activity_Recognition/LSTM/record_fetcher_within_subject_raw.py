import numpy as np
import glob
import random
import sys

class record_fetcher(object):
    '''
    creates feature arrays and labels from raw accelerometer/demographic data
    no test/train splitting is included
    
    methods:
      - fetch(batch_size,binary)
        separates raw accelerometer/demographic data into windows and creates
        input features and labels for conv-lstm classifier
        parameters:
          - batch_size: integer
            number of frames to use for each train/test instance
            e.g. 1000 means each test/train instance represents 10 seconds of data
          - binary: boolean (default True)
            use True to set labels for ambulatory/non-ambulatory
            use False to set labels for non-ambulatory/walking/running/upstairs/downstairs
        outputs:
          - numpy array representing raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
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
        separates raw accelerometer/demographic data into windows and creates
        input features and labels for conv-lstm classifier
        parameters:
          - batch_size: integer
            number of frames to use for each train/test instance
            e.g. 1000 means each test/train instance represents 10 seconds of data
          - binary: boolean (default True)
            use True to set labels for ambulatory/non-ambulatory
            use False to set labels for non-ambulatory/walking/running/upstairs/downstairs
        outputs:
          - numpy array representing raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
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
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([0,1]))
                    
            for a in self.nonambulatory:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([1,0]))

        #for non-ambulatory/walking/running/upstairs/downstairs classification
        else:
            for a in self.nonambulatory:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([1,0,0,0,0]))
                    
            for a in self.walking:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([0,1,0,0,0]))
                    
            for a in self.running:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([0,0,1,0,0]))
                    
            for a in self.upstairs:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([0,0,0,1,0]))
                    
            for a in self.downstairs:
                print 'processing %s' % a
                array = np.load(a)
                
                #avoid arrays smaller than batch_size
                if array.shape[0] <= batch_size:
                    continue
    
                #separate array into batches
                seg = array.shape[0]//batch_size
                for i in range(seg):
                    batch = array[i*batch_size:(i+1)*batch_size,:]
                    X_list.append(np.expand_dims(batch,0))
                
                    #create label
                    y_list.append(np.array([0,0,0,0,1]))
            
        #pair X/y together and shuffle
        print 'shuffling records'
        Xy = zip(X_list,y_list)
        random.shuffle(Xy)

        #separate X from y
        X = np.array([record[0] for record in Xy])
        y = np.array([record[1] for record in Xy])
        
        print 'feature vector shape:', X.shape
        print 'label vector shape:', y.shape
        
        return X, y
        
if __name__ == "__main__":

    # verify the required arguments are given
    if (len(sys.argv) < 2):
        print 'Usage: python record_fetcher_within_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)
    
    if sys.argv[1] == '1':
        binary = True
    elif sys.argv[1] == '0':
        binary = False
    else:
        print 'Usage: python record_fetcher_within_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)

    rf = record_fetcher()
    X,y = rf.fetch(1000,binary=binary)
    np.save('X_raw',X)
    np.save('y_raw',y)