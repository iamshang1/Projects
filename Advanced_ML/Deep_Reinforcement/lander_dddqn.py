import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

class agent(object):
    def __init__(self,state_size=8,actions=4,gamma=0.99,replay=int(1e6)):
    
        self.replay_memory = deque(maxlen=replay)
        self.decay_len = 8e4
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_step = (1.0-self.epsilon_min)/self.decay_len
        self.total_steps = 0
        self.gamma = gamma
        
        self.state = tf.placeholder(dtype=tf.float32, shape=[None,state_size])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None])
        batch_size = tf.reduce_sum(tf.ones_like(self.action))
    
        with tf.variable_scope('fast'):
            self.hidden = tf.layers.dense(self.state,512,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value_dense = tf.layers.dense(self.hidden,256,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value = tf.layers.dense(self.value_dense,1,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.adv_dense = tf.layers.dense(self.hidden,512,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.advantage = tf.layers.dense(self.adv_dense,n_actions,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q_values = tf.squeeze(self.value + tf.subtract(
                            self.advantage,tf.reduce_mean(self.advantage,axis=1,keepdims=True)))
            
        with tf.variable_scope('slow'):
            self.hidden_ = tf.layers.dense(self.state,512,activation=tf.nn.elu,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value_dense_ = tf.layers.dense(self.hidden_,256,activation=tf.nn.elu,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value_ = tf.layers.dense(self.value_dense_,1,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.adv_dense_ = tf.layers.dense(self.hidden_,512,activation=tf.nn.elu,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.advantage_ = tf.layers.dense(self.adv_dense_,n_actions,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.Q_values_ = tf.squeeze(self.value_ + tf.subtract(
                            self.advantage_,tf.reduce_mean(self.advantage_,axis=1,keepdims=True)))
                        
        self.Q_pred = tf.gather_nd(self.Q_values, tf.stack((tf.range(batch_size),self.action), axis=1))
        self.loss = tf.reduce_mean(tf.losses.huber_loss(self.target,self.Q_pred))
        self.fast_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='fast')
        self.slow_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='slow')
        self.optimizer = tf.train.AdamOptimizer(5e-5).minimize(self.loss,var_list=self.fast_vars)
        
        self.sess = tf.Session()	
        self.sess.run(tf.global_variables_initializer())
    
    def add_memory(self,observation,action,reward,next_observation,done):
    
        self.replay_memory.append((observation, action, reward, next_observation,done))
                            
    def update_slow(self):
    
        update_slow_target_ops = []
        for slow_var,fast_var in zip(self.slow_vars,self.fast_vars):
            update_slow_target_op = slow_var.assign(fast_var)
            update_slow_target_ops.append(update_slow_target_op)
        update_slow_target_op = tf.group(*update_slow_target_ops)
        
        self.sess.run(update_slow_target_op)
    
    def train(self,batch_size=1024):
    
        if len(self.replay_memory) < batch_size:
            return 0,self.epsilon
    
        if self.total_steps % 100 == 0:
            self.update_slow()
    
        #sample memory
        batch = random.sample(self.replay_memory,batch_size)
        states = np.asarray([elem[0] for elem in batch])
        actions = np.asarray([elem[1] for elem in batch])
        rewards = np.asarray([elem[2] for elem in batch])
        next_states = np.asarray([elem[3] for elem in batch])
        terminal = np.asarray([elem[4] for elem in batch])
    
        #calculate target rewards
        d = {self.state:next_states}
        action_v,q_v = self.sess.run([self.Q_values,self.Q_values],feed_dict=d)
        action_v = np.argmax(action_v,1)
        next_q = q_v[range(batch_size),action_v]
        targets = rewards + terminal * self.gamma * next_q
        
        #train
        d = {self.state:states,self.action:actions,self.target:targets}
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=d)
        
        self.total_steps += 1
        if self.total_steps < self.decay_len:
            self.epsilon -= self.epsilon_step
        
        return loss,self.epsilon
        
    def predict(self,state):
    
        if np.random.random() < self.epsilon:
            action = np.random.randint(n_actions)
        else:
            Q_pred = self.sess.run(self.Q_values,feed_dict={self.state:np.expand_dims(state,0)})
            action = np.argmax(np.squeeze(Q_pred))
        return action
    

#init environment
env = gym.make('LunarLander-v2')
state_dim = np.prod(np.array(env.observation_space.shape))
n_actions = env.action_space.n
dqn_agent = agent(state_size=state_dim,actions=n_actions)

for ep in range(5000):
    total_reward = 0
    steps_in_ep = 0
    observation = env.reset()
    
    for t in range(1000):
        mean_loss = []
        action = dqn_agent.predict(observation)
        
        next_observation, reward, done, _info = env.step(action)
        env.render('human')
        total_reward += reward
        reward /= 10
        dqn_agent.add_memory(observation, action, reward, next_observation,0.0 if done else 1.0)
        loss,epsilon = dqn_agent.train()
        mean_loss.append(loss)
            
        observation = next_observation
        steps_in_ep += 1
        
        if done: 
            break
            
    mean_loss = np.mean(mean_loss)
    print('Episode %2i, Reward: %6.2f, Steps: %4i, Loss: %5.4f, Next eps: %4.3f' % 
         (ep,total_reward,steps_in_ep,mean_loss,epsilon))

env.close()