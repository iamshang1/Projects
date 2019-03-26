import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

class agent(object):
    def __init__(self,state_size=8,actions=4,gamma=0.99):
        '''
        advantage actor critic for lunar lander
        '''
        
        self.episode_memory = []             #store all observations from latest episode
        self.gamma = gamma
        
        #network inputs
        self.state = tf.placeholder(dtype=tf.float32, shape=[None,state_size])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None])
        self.value_target = tf.placeholder(dtype=tf.float32, shape=[None])
        batch_size = tf.reduce_sum(tf.ones_like(self.action))
        
        #actor network predicts actions for given states
        self.actor_hidden1 = tf.layers.dense(self.state,128,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actor_hidden2 = tf.layers.dense(self.actor_hidden1,128,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.actor_logits = tf.layers.dense(self.actor_hidden2,n_actions,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.action_probs = tf.nn.softmax(self.actor_logits)

        #critic network calculates values of given states
        self.critic_hidden1 = tf.layers.dense(self.state,128,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.critic_hidden2 = tf.layers.dense(self.critic_hidden1,64,activation=tf.nn.elu,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.value = tf.layers.dense(self.critic_hidden2,1,
                        kernel_initializer=tf.contrib.layers.xavier_initializer())
             
        #actor loss is calculated using REINFORCE
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actor_logits,labels=self.action)
        self.actor_loss = tf.reduce_mean(neg_log_prob*self.advantage)
        
        #critic loss is temporal difference error similar to DQN
        self.critic_loss = tf.reduce_mean(tf.losses.huber_loss(tf.expand_dims(self.value_target,1),self.value))
        
        #optimizers
        self.optimizer_actor = tf.train.AdamOptimizer(0.001).minimize(self.actor_loss)
        self.optimizer_critic = tf.train.AdamOptimizer(0.0001).minimize(self.critic_loss) 
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    
    def add_memory(self,observation,action,reward,next_observation,done):
        '''
        add experience to episode memory
        '''
    
        self.episode_memory.append((observation, action, reward, next_observation,done))
    
    def train(self):
        '''
        train model using samples from episode memory
        '''
    
        #create batch from episode memory
        states = np.asarray([elem[0] for elem in self.episode_memory])
        actions = np.asarray([elem[1] for elem in self.episode_memory])
        rewards = np.asarray([elem[2] for elem in self.episode_memory])
        next_states = np.asarray([elem[3] for elem in self.episode_memory])
        terminal = np.asarray([elem[4] for elem in self.episode_memory])
    
        #calculate value of next states
        d = {self.state:next_states}
        value_next = self.sess.run(self.value,feed_dict=d)
        value_next = np.squeeze(value_next)
        td_targets = rewards + terminal * self.gamma * value_next
        
        #advantage of given action is next value - current value
        d = {self.state:states}
        value_now = self.sess.run(self.value,feed_dict=d)
        value_now = np.squeeze(value_now)
        td_errors = td_targets - value_now
        
        #train
        d = {self.state:states,self.action:actions,self.advantage:td_errors,self.value_target:td_targets}
        actor_loss,critic_loss,_,_ = self.sess.run([self.actor_loss,self.critic_loss,
                                     self.optimizer_actor,self.optimizer_critic],feed_dict=d)
        
        #clear episode memory        
        self.episode_memory = []
        
        return actor_loss,critic_loss
        
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
ac2_agent = agent(state_size=state_dim,actions=n_actions)
viz = False

#run episodes
for ep in range(5000):
    total_reward = 0
    steps_in_ep = 0
    observation = env.reset()
    
    #run each episode for 1000 steps
    for t in range(1000):
    
        #get next action
        action = ac2_agent.predict(observation)
        next_observation, reward, done, _info = env.step(action)
        total_reward += reward
        
        #visualize
        if viz:
            env.render('human')

        #normalize reward
        reward /= 100
        
        #add obseravtion to memory
        ac2_agent.add_memory(observation, action, reward, next_observation,0.0 if done else 1.0)
        observation = next_observation
        steps_in_ep += 1
        
        #terminate episode if done
        if done: 
            break
    
    #train network
    actor_loss,critic_loss = ac2_agent.train()
    print('Episode %2i, Reward: %6.2f, Steps: %4i, Actor Loss: %.4f, Critic Loss: %.4f' % 
                (ep,total_reward,steps_in_ep,actor_loss,critic_loss))

    #enable vizualization once model gets over 200 score
    if total_reward >= 200:
        viz = True
    
env.close()