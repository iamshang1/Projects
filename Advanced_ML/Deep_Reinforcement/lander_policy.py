import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

class agent(object):
    def __init__(self,state_size=8,actions=4,gamma=0.99):
        '''
        policy-based deep reinforcement learning for lunar lander
        '''
    
        self.episode_memory = []             #store all observations from latest episode
        self.gamma = gamma
        
        #network inputs
        self.state = tf.placeholder(dtype=tf.float32, shape=[None,state_size])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None])
        
        #neural network architecture
        self.hidden1 = tf.layers.dense(self.state,128,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.hidden2 = tf.layers.dense(self.hidden1,128,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.layers.dense(self.hidden2,n_actions,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.action_probs = tf.nn.softmax(self.logits)
        
        #train using REINFORCE algorithm
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.action)
        self.loss = tf.reduce_mean(neg_log_prob*self.advantage)        
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)        
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def add_memory(self,observation,action,reward,next_observation,done):
        '''
        add experience to episode memory
        '''
        
        self.episode_memory.append((observation, action, reward, next_observation,done))
    
    def calculate_inputs(self):
        '''
        create training batch from episode memory
        '''
        
        states = np.asarray([elem[0] for elem in self.episode_memory])
        actions = np.asarray([elem[1] for elem in self.episode_memory])
        rewards = np.asarray([elem[2] for elem in self.episode_memory])
        next_states = np.asarray([elem[3] for elem in self.episode_memory])
        terminal = np.asarray([elem[4] for elem in self.episode_memory])
    
        #calculate discounted rewards for every step in the episode
        discounted_episode_rewards = np.zeros_like(rewards)
        cumulative = 0
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[t]
            discounted_episode_rewards[t] = cumulative
        
        return states,actions,discounted_episode_rewards
    
    def train(self):
        '''
        train model using samples from episode memory
        '''
        
        #create batch from episode memory
        states,actions,advantage = self.calculate_inputs()
        
        #train
        d = {self.state:states,self.action:actions,self.advantage:advantage}
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=d)
        
        #clear episode memory
        self.episode_memory = []
        
        return loss
        
    def predict(self,state):
        '''
        predict next action given an input state
        '''
    
        #next action is sampled using softmax probabilities assigned to each action
        action_probs = self.sess.run(self.action_probs,feed_dict={self.state:np.expand_dims(state,0)})
        action_probs = np.squeeze(action_probs)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        return action
    
    def save(self,filename):
        '''
        save network params
        '''

        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load network params
        '''
        
        self.saver.restore(self.sess,filename)
        

#init environment
env = gym.make('LunarLander-v2')
state_dim = np.prod(np.array(env.observation_space.shape))
n_actions = env.action_space.n
pol_agent = agent(state_size=state_dim,actions=n_actions)
viz = False

#run episodes
for ep in range(5000):
    total_reward = 0
    steps_in_ep = 0
    observation = env.reset()
    
    #run each episode for 1000 steps
    for t in range(1000):
    
        #get next action
        action = pol_agent.predict(observation)
        next_observation, reward, done, _info = env.step(action)
        total_reward += reward
        
        #visualize
        if viz:
            env.render('human')
        
        #normalize reward
        reward /= 100
        
        #add obseravtion to memory
        pol_agent.add_memory(observation, action, reward, next_observation,0.0 if done else 1.0)
        observation = next_observation
        steps_in_ep += 1
        
        #terminate episode if done
        if done:
            break
    
    #train network
    loss = pol_agent.train()
    print('Episode %2i, Reward: %6.2f, Steps: %4i, Loss: %.4f' % 
                (ep,total_reward,steps_in_ep,loss))

    #enable vizualization once model gets over 200 score
    if total_reward >= 200:
        viz = True
                
env.close()