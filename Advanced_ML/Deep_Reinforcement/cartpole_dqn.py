import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import gym
from skimage import transform 
from skimage.color import rgb2gray 
from collections import deque
import random

class agent(object):
    def __init__(self,action_size=2,gamma=0.95,memory=10000):
    
        #params
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.001             # exponential decay rate for exploration prob
        self.gamma = gamma
        self.action_size = action_size
        self.decay_step = 0
    
        #init experience replay memory
        self.buffer = deque(maxlen=memory)   
    
        #build DQN layers
        self.state = tf.placeholder(tf.float32,shape=(None,4))
        self.action = tf.placeholder(tf.float32,shape=(None,2))
        self.target = tf.placeholder(tf.float32,shape=(None))
        self.hidden1 = tf.layers.dense(self.state,24,activation=tf.nn.elu,
                       kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.hidden2 = tf.layers.dense(self.hidden1,24,activation=tf.nn.elu,
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
        
    def predict(self,stacked_state,explore=True):
        r = np.random.rand()
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop) \
                       * np.exp(-self.decay_rate * self.decay_step)
        
        if explore and (explore_prob > r):
            action = np.random.randint(0,self.action_size)
        else:
            d = {self.state:np.expand_dims(stacked_state,0)}
            Qs = self.sess.run(self.output,feed_dict=d)
            action = np.argmax(Qs)
                  
        return action,explore_prob
        
    def train(self,batch_size=128):
    
        self.decay_step +=1
        
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
        self.buffer.append(experience)
    
    def sample_memory(self,batch_size):
        buffer_size = len(self.buffer)
        idx = np.random.choice(np.arange(buffer_size),size=batch_size,replace=False)
        return [self.buffer[i] for i in idx]
        
    def save(self,filename):

        self.saver.save(self.sess,filename)

    def load(self,filename):

        self.saver.restore(self.sess,filename)
        

env = gym.make('CartPole-v1')
action_size = env.action_space.n
DQagent = agent(action_size=action_size)

for ep in range(50000):
    print('training episode %i' % (ep+1))
    state = env.reset()
    done = False
    t = 0
    episode_rewards = []
    mean_loss = []
    
    explore_prob = 1.0
    
    while not done and t < 200:
    
        t += 1
    
        if len(DQagent.buffer) < 1000:
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            reward /= 5
            
            if done:
                reward = -1
            
            DQagent.add_memory((state,action,reward,new_state,done))
            state = new_state

        else:

            action,explore_prob = DQagent.predict(state)
            new_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            reward /= 5

            if done:
                reward = -1
                
            DQagent.add_memory((state,action,reward,new_state,done))
            state = new_state
            mean_loss.append(DQagent.train())
        
    # Get the total reward of the episode
    loss = np.mean(mean_loss)
    total_reward = np.sum(episode_rewards)
    print('Episode: {}'.format(ep+1),
          'Total reward: {}'.format(total_reward),
          'Explore P: {:.4f}'.format(explore_prob),
          'Training Loss {:.4f}'.format(loss))
                  
    if ep % 10 == 0:
        DQagent.save("savedmodels/cartpole.ckpt")
'''
#show trained agent
DQagent.load("savedmodels/cartpole.ckpt")
for ep in range(10):
    state = env.reset()
    done = False
    t = 0
    
    while not done and t < 10000:
    
        t += 1
        action,explore_prob = DQagent.predict(state,explore=False)
        state, reward, done, info = env.step(action)
        env.render('human')
'''
env.close()