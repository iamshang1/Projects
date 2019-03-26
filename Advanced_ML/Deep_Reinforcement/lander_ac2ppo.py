import numpy as np
import gym
import tensorflow as tf
import random
from collections import deque

class agent(object):
    def __init__(self,state_size=8,actions=4,gamma=0.99):
    
        self.replay_memory = []
        self.gamma = gamma
        
        self.state = tf.placeholder(dtype=tf.float32, shape=[None,state_size])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None])
        self.value_target = tf.placeholder(dtype=tf.float32, shape=[None])
        batch_size = tf.reduce_sum(tf.ones_like(self.action))
        
        with tf.variable_scope('actor'):
            self.actor_hidden1 = tf.layers.dense(self.state,128,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.actor_hidden2 = tf.layers.dense(self.actor_hidden1,128,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.actor_logits = tf.layers.dense(self.actor_hidden2,n_actions,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.action_probs = tf.nn.softmax(self.actor_logits)
            
        with tf.variable_scope('actor_old'):
            self.actor_hidden1_ = tf.layers.dense(self.state,128,activation=tf.nn.elu,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.actor_hidden2_ = tf.layers.dense(self.actor_hidden1_,128,activation=tf.nn.elu,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.actor_logits_ = tf.layers.dense(self.actor_hidden2_,n_actions,trainable=False,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.action_probs_ = tf.nn.softmax(self.actor_logits_)

        with tf.variable_scope('critic'):
            self.critic_hidden1 = tf.layers.dense(self.state,128,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.critic_hidden2 = tf.layers.dense(self.critic_hidden1,64,activation=tf.nn.elu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.value = tf.layers.dense(self.critic_hidden2,1,
                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='actor')
        self.actor_old_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='actor_old')
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='critic')
        
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actor_logits,labels=self.action)
        ratio = neg_log_prob/tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.actor_logits_,labels=self.action)
        surr = neg_log_prob * self.advantage
        actor_loss = tf.reduce_mean(tf.minimum(surr,tf.clip_by_value(ratio,0.8,1.2)*self.advantage))
        critic_loss = tf.reduce_mean(tf.losses.huber_loss(tf.expand_dims(self.value_target,1),self.value))
        self.loss = actor_loss + critic_loss
        
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)        
        
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.cur_params = self.sess.run(self.actor_vars)
    
    def add_memory(self,observation,action,reward,next_observation,done):
    
        self.replay_memory.append((observation, action, reward, next_observation,done))

    def update_slow(self):
    
        self.prev_params = self.cur_params[:]
        self.cur_params = self.sess.run(self.actor_vars)
    
        update_slow_target_ops = []
        for slow_var,fast_var in zip(self.actor_old_vars,self.prev_params):
            update_slow_target_op = slow_var.assign(fast_var)
            update_slow_target_ops.append(update_slow_target_op)
        update_slow_target_op = tf.group(*update_slow_target_ops)
        
        self.sess.run(update_slow_target_op)
    
    def train(self):
    
        self.update_slow()
    
        states = np.asarray([elem[0] for elem in self.replay_memory])
        actions = np.asarray([elem[1] for elem in self.replay_memory])
        rewards = np.asarray([elem[2] for elem in self.replay_memory])
        next_states = np.asarray([elem[3] for elem in self.replay_memory])
        terminal = np.asarray([elem[4] for elem in self.replay_memory])
    
        #calculate target rewards
        d = {self.state:next_states}
        value_next = self.sess.run(self.value,feed_dict=d)
        value_next = np.squeeze(value_next)
        td_targets = rewards + terminal * self.gamma * value_next
        
        d = {self.state:states}
        value_now = self.sess.run(self.value,feed_dict=d)
        value_now = np.squeeze(value_now)
        td_errors = td_targets - value_now
        
        #train
        d = {self.state:states,self.action:actions,self.advantage:td_errors,self.value_target:td_targets}
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict=d)
                    
        self.replay_memory = []
        
        return loss
        
    def predict(self,state):
    
        action_probs = self.sess.run(self.action_probs,feed_dict={self.state:np.expand_dims(state,0)})
        action_probs = np.squeeze(action_probs)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        
        return action
    
    def save(self,filename):

        self.saver.save(self.sess,filename)

    def load(self,filename):

        self.saver.restore(self.sess,filename)
        self.cur_params = self.sess.run(self.actor_vars)
        self.prev_params = self.cur_params[:]

#init environment
env = gym.make('LunarLander-v2')
state_dim = np.prod(np.array(env.observation_space.shape))
n_actions = env.action_space.n
ac2_agent = agent(state_size=state_dim,actions=n_actions)
#ac2_agent.load("savedmodels/lander.ckpt")
total_steps = 0

for ep in range(5000):
    total_reward = 0
    steps_in_ep = 0
    observation = env.reset()
    
    for t in range(1000):
        action = ac2_agent.predict(observation)
        
        next_observation, reward, done, _info = env.step(action)
        #env.render('human')
        total_reward += reward
        reward /= 100
        ac2_agent.add_memory(observation, action, reward, next_observation,0.0 if done else 1.0)
        observation = next_observation
        steps_in_ep += 1
        total_steps += 1
        
        if done: 
            break
            
    loss = ac2_agent.train()
    print('Episode %2i, Reward: %6.2f, Steps: %4i, Loss: %.4f' % 
                (ep,total_reward,steps_in_ep,loss))

    if ep % 100 == 0:
        ac2_agent.save("savedmodels/lander.ckpt")
                
env.close()