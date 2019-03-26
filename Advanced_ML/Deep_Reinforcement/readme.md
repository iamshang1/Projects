**Under Construction**

#Deep Reinforcement Learning

This project inludes several exercises applying variations of deep reinforcement learning on OpenAI Gym environments.

*insert gif images*

###Deep Q-Networks

In Q-learning, given a state, each action possible from that state is assigned a Q-value that indicates the potential value of taking that action from the state; these Q-values are learned over time using a method such as temporal difference learning (see [basic Q-learning project](https://github.com/iamshang1/Projects/tree/master/Basic_ML/Reinforcement_Learning)). In traditional Q-learning, the Q-values for every possible state are stored in a table, which is intractable for problems with large state/action spaces. In deep Q-learning, each state is represented as a set of input features, and the Q-values for each possible action are then calculated using a neural network; this eliminates the need to store the Q-values for every possible state/actions in memory.

###Double Deep Q-Networks

Deep Q-learning networks can be difficult to train because . One way to mitigate this is to use two networks in parallel -- a fast network and a slow network.

###Dueling Deep Q-Networks

###Policy Networks

unlike DQN, trains once per episode
no replay memory, just trains on observations from previous episode

###Actor Critic Networks

like policy networks, trains once per episode

###Notes on Running Code

 - cartpole_dqn.py trains in about 200 episodes (including burn-in)
 - lander_dddqn.py trains in about 500 episodes (including burn-in)