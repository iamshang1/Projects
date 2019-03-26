import numpy as np
import tensorflow as tf
import gym
from collections import deque
import random

class agent(object):
    def __init__(self,action_size=2,gamma=0.95,memory=10000):
        '''
        simple deep Q learning network for cartpole
        '''
    
        #params
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.001             # exponential decay rate for exploration prob
        self.gamma = gamma
        self.action_size = action_size
        self.total_steps = 0
    
        #init experience replay memory
        self.buffer = deque(maxlen=memory)
        self.burn_in = 2000                 # number of observations before training
    
        #build DQN layers
        self.state = tf.placeholder(tf.float32,shape=(None,4))
        self.action = tf.placeholder(tf.float32,shape=(None,2))
        self.target = tf.placeholder(tf.float32,shape=(None))
        self.hidden1 = tf.layers.dense(self.state,36,activation=tf.nn.elu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.hidden2 = tf.layers.dense(self.hidden1,36,activation=tf.nn.elu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.output = tf.layers.dense(self.hidden2,2,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.Q_pred = tf.reduce_sum(tf.multiply(self.output,self.action),1)
        
        #optimizer
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.target,self.Q_pred))
        self.optimizer = tf.train.AdamOptimizer(0.0001,0.9,0.99).minimize(self.loss)
        
        #init op
        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        
    def predict(self,stacked_state):
        '''
        predict next action given an input state
        '''
    
        #calculate explore probability
        r = np.random.rand()
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop) \
                       * np.exp(-self.decay_rate * self.total_steps)
        
        #explore
        if explore_prob > r:
            action = np.random.randint(0,self.action_size)
        
        #exploit
        else:
            d = {self.state:np.expand_dims(stacked_state,0)}
            Qs = self.sess.run(self.output,feed_dict=d)
            action = np.argmax(Qs)
                  
        return action,explore_prob
        
    def train(self,batch_size=256):
        '''
        train model using samples from replay memory
        '''
    
        #fill up memory replay before training
        if len(self.buffer) < self.burn_in:
            return 0
    
        self.total_steps +=1
        
        #get batch
        batch = self.sample_memory(batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        actions_onehot = [np.zeros(self.action_size) for i in range(len(batch))]
        rewards = np.array([each[2] for each in batch]) 
        next_states = np.array([each[3] for each in batch])
        dones = np.array([each[4] for each in batch])
        
        #calculate target Q values
        d = {self.state:next_states}
        Qs_next_state = self.sess.run(self.output,feed_dict=d)
        
        target_Qs_batch = []
        for i in range(0, len(batch)):
            actions_onehot[i][actions[i]] = 1
            if dones[i]:
                target_Qs_batch.append(rewards[i])
            else:
                target = rewards[i] + self.gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
        actions_onehot = np.array(actions_onehot)
        
        #train
        targets = np.array([each for each in target_Qs_batch])
        d = {self.state:states,self.target:targets,self.action:actions_onehot}
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=d)
        return loss
        
    def add_memory(self,experience):
        '''
        add experience to replay memory
        '''
        
        self.buffer.append(experience)
    
    def sample_memory(self,batch_size):
        '''
        create training batch from replay memory
        '''
    
        buffer_size = len(self.buffer)
        idx = np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)
        return [self.buffer[i] for i in idx]
        
    def save(self,filename):
        self.saver.save(self.sess,filename)

    def load(self,filename):
        self.saver.restore(self.sess,filename)
        
#initialize environment
env = gym.make('CartPole-v1')
action_size = env.action_space.n
DQagent = agent(action_size=action_size)
viz = False

#run episodes
for ep in range(500):
    state = env.reset()
    done = False
    t = 0
    episode_rewards = []
    mean_loss = []
    explore_prob = 1.0
    
    #terminate episode when done
    while not done and t < 200:
    
        t += 1

        #get next action
        action,explore_prob = DQagent.predict(state)
        new_state, reward, done, info = env.step(action)
        episode_rewards.append(reward)
        
        #visualize
        if viz:
            env.render('human')
        
        #normalize rewards
        reward /= 5
        if done:
            reward = -1
            
        #update memory and train
        DQagent.add_memory((state,action,reward,new_state,done))
        state = new_state
        mean_loss.append(DQagent.train())
        
    #show episode results
    loss = np.mean(mean_loss)
    total_reward = np.sum(episode_rewards)
    print('Episode: {}'.format(ep+1),
          'Total reward: {}'.format(total_reward),
          'Explore P: {:.4f}'.format(explore_prob),
          'Training Loss {:.4f}'.format(loss))
          
    #enable vizualization once model gets a perfect score
    if total_reward >= 200:
        viz = True

env.close()