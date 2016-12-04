import numpy as np
import glob
import random
import sys

class record_fetcher(object):
    '''
    creates feature arrays and labels from raw accelerometer/demographic data
    splits features and labels between subjects into test/train sets
    
    methods:
      - fetch(batch_size,binary,seed)
        separates raw accelerometer/demographic data into windows and creates
        input features and labels for conv-lstm classifier
        parameters:
          - batch_size: integer
            number of frames to use for each train/test instance
            e.g. 1000 means each test/train instance represents 10 seconds of data
          - binary: boolean (default True)
            use True to set labels for ambulatory/non-ambulatory
            use False to set labels for non-ambulatory/walking/running/upstairs/downstairs
          - seed: integer (default None)
            (optional) seed to use for random test/train splitting
        outputs:
          - numpy array representing training raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
          - numpy array representing testing raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
          - numpy array representing training activity label over each time window
          - numpy array representing testing activity label over each time window
    '''

    def __init__(self):

        #collect all valid subject ids
        self.subjects = [102,103,105,106,107,108,110,112,113,114,115,116,117,118,119,120,\
            121,122,123,124,125,126,127,128,129,130,131,132,133,134,136,137,138,139,140,\
            142,143,144,146,148,149,150,151,152,153,154,155,156,157,159,160,161,162,163,\
            164,165,166,169,170,171,172,173,174,175,177,178,179,180,181,182,183,184,185,\
            186,187,188,189,190,191,192]
    
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
        
    def fetch(self,batch_size,binary=True,seed=None):
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
          - seed: integer (default None)
            (optional) seed to use for random test/train splitting
        outputs:
          - numpy array representing training raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
          - numpy array representing testing raw accelerometer and demographic data
            dimension 0 is the index of the batch window
            dimension 1 is the number of filter maps (1 filter map per instance)
            dimension 2 is the time frame of the accelerometer data within the batch window
            dimension 3 is the raw accelerometer and demographic data for the specified time frame
          - numpy array representing training activity label over each time window
          - numpy array representing testing activity label over each time window
        '''
        np.random.seed(seed)
        X_test_subjects = np.random.choice(self.subjects,6) 
        
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([0,1]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([0,1]))
                    
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([1,0]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([1,0]))

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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([1,0,0,0,0]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([1,0,0,0,0]))
                    
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([0,1,0,0,0]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([0,1,0,0,0]))
                    
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([0,0,1,0,0]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([0,0,1,0,0]))
                    
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([0,0,0,1,0]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([0,0,0,1,0]))
                    
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
                    
                    #if subject id in test set, save features and labels to test set
                    if int(a[15:18]) in X_test_subjects:
                        X_test_list.append(np.expand_dims(batch,0))
                        y_test_list.append(np.array([0,0,0,0,1]))
                    
                    #else save features and labels to train set
                    else:
                        X_train_list.append(np.expand_dims(batch,0))
                        y_train_list.append(np.array([0,0,0,0,1]))
            
        #pair training X/y together and shuffle
        print 'shuffling records'
        Xy = zip(X_train_list,y_train_list)
        random.shuffle(Xy)

        #separate training X from y
        X_train = np.array([record[0] for record in Xy])
        y_train = np.array([record[1] for record in Xy])
        
        print 'feature vector shape:', X_train.shape
        print 'label vector shape:', y_train.shape
        
        #pair testing X/y together and shuffle
        Xy = zip(X_test_list,y_test_list)
        random.shuffle(Xy)

        #separate testing X from y
        X_test = np.array([record[0] for record in Xy])
        y_test = np.array([record[1] for record in Xy])
        
        print 'feature vector shape:', X_test.shape
        print 'label vector shape:', y_test.shape
        
        return X_train, X_test, y_train, y_test
        
if __name__ == "__main__":

    # verify the required arguments are given
    if (len(sys.argv) < 2):
        print 'Usage: python record_fetcher_between_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)
    
    if sys.argv[1] == '1':
        binary = True
    elif sys.argv[1] == '0':
        binary = False
    else:
        print 'Usage: python record_fetcher_between_subject_raw.py <1 for 2-category labels, 0 for 5-category labels>'
        exit(1)
        
    rf = record_fetcher()
    X_train,X_test,y_train,y_test = rf.fetch(1000,binary=binary,seed=1)
    np.save('X_train_raw',X_train)
    np.save('X_test_raw',X_test)
    np.save('y_train_raw',y_train)
    np.save('y_test_raw',y_test)