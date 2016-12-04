import numpy as np
import glob
import sys
import random
import matplotlib.pyplot as plt

class visualizer(object):
    '''
    plots x, y, and z axis of accelerometer data for a selected activity
    
    methods:
      - visualize(batch_size,activity)
        randomly selects a window of time of a specific activity and plots the
        x, y, and z accelerometer readings for that window
        parameters:
          - batch_size: integer
            number of frames in time window
          - activity: string
            activity of time window, valid options are 'nonambulatory',
            'walking', 'running', 'upstairs', and 'downstairs'
    '''
    def __init__(self):
        
        #categorize activity ids
        self.dic = {
        'nonambulatory': [19,20,21,22],
        'walking': [11,12,13,14,23,24,25,26,27,28,29,30,31,32],
        'running': [16,17,18],
        'upstairs': [33],
        'downstairs': [34]
        }

        #get filenames for all activity arrays
        self.nonambulatory = []
        for i in self.dic['nonambulatory']:
            self.nonambulatory.extend(glob.glob('./data/arrays/*_%i_*' % i))
            
        self.walking = []
        for i in self.dic['walking']:
            self.walking.extend(glob.glob('./data/arrays/*_%i_*' % i))
            
        self.running = []
        for i in self.dic['running']:
            self.running.extend(glob.glob('./data/arrays/*_%i_*' % i))
            
        self.upstairs = []
        for i in self.dic['upstairs']:
            self.upstairs.extend(glob.glob('./data/arrays/*_%i_*' % i))
            
        self.downstairs = []
        for i in self.dic['downstairs']:
            self.downstairs.extend(glob.glob('./data/arrays/*_%i_*' % i))
        
    def visualize(self,batch_size,activity=None):
        '''
        randomly selects a window of time of a specific activity and plots the
        x, y, and z accelerometer readings for that window
        
        parameters:
          - batch_size: integer
            number of frames in time window
          - activity: string
            activity of time window, valid options are 'nonambulatory',
            'walking', 'running', 'upstairs', and 'downstairs'
        '''
        if activity == 'nonambulatory':
            activity_list = self.nonambulatory
        elif activity == 'walking':
            activity_list = self.walking
        elif activity == 'running':
            activity_list = self.running
        elif activity == 'upstairs':
            activity_list = self.upstairs
        elif activity == 'downstairs':
            activity_list = self.downstairs
        
        #load random activity array from selected category
        choice = random.choice(activity_list)
        array = np.load(choice)
        
        #if selected activity array is shorter than window, reload
        while array.shape[0] <= batch_size:
            choice = random.choice(activity_list)
            array = np.load(choice)
        
        #select random window in activity array
        start = np.random.randint(0, array.shape[0]-batch_size)
        batch = array[start:start+batch_size,:3]
        
        #plot and save
        plt.plot(batch)
        plt.savefig(activity+'.png')
        plt.show()
        
if __name__ == "__main__":

    # verify the required arguments are given
    if (len(sys.argv) < 2):
        print 'Usage: python visualization.py <"nonambulatory", "walking", "running", "upstairs", or "downstairs">'
        exit(1)

    activity = sys.argv[1]
        
    vs = visualizer()
    vs.visualize(1000,activity)