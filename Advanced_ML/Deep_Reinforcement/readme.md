**Under Construction**

# Exercises in Deep Reinforcement Learning

This project inludes several exercises applying variations of deep reinforcement learning on OpenAI Gym environments.

*insert gif images*

### Deep Q-Networks

In Q-learning, given a state, each action possible from that state is assigned a Q-value that indicates the potential value of taking that action from the state; these Q-values are learned over time using a method such as temporal difference learning (see [basic Q-learning project](https://github.com/iamshang1/Projects/tree/master/Basic_ML/Reinforcement_Learning)). In traditional Q-learning, the Q-values for every possible state are stored in a table, which is intractable for problems with large state/action spaces. In deep Q-learning, each state is represented as a set of input features, and the Q-values for each possible action are then calculated using a neural network; this eliminates the need to store the Q-values for every possible state/actions in memory.

### Double Deep Q-Networks

Deep Q-learning networks are generally trained using temporal difference learning. As a result, basic DQNs can be difficult to train because the target Q-values used for training are calculated using the same network that is being trained. In other words, each update made to the network directly effects the calculations of the target Q-values used for the following update,  so the network is always chasing a moving target. One way to mitigate this is to use two networks in parallel -- a fast network and a slow network. The fast network is updated at every step using the target Q-values calculated by the slow network, which is updated once every *X* steps by copying over the parameters from the fast network.

### Dueling Deep Q-Networks

A potential inefficiency of DQNs is that the network must learn an independent Q-value for each possible action. For problems with large action spaces, this means that the network must explore all possible actions before it will learn the appropriate Q-values for each action. One way to mitigate this is to separate the calculation of the value of each state from the calculation of the relative advantage of each action; the final Q-value is then calculated as the value of the state plus the relative advantage of the selected action. Therefore, the network will know something about the Q-value of an action (based off the value of the state) even if it has not fully explored that action.

### Policy Networks

Instead of , predict action directly. More stable training and no need for a schedule (required for DQN-based networks); however, policy networks are more likely to get trapped in local optima.

Whereas the DQN-based approaches make an update every step in the episode, our policy network trains only once at the end of each episode. In addition, DQN-based approaches often utilize a replay memory in which . On the other hand, our policy network does not utilize a replay memory; instead, each training batch is simply composed of the observations from latest episode.

### Actor Critic Networks

like policy networks, trains once per episode

### Notes on Running Code

 - cartpole_dqn.py trains in about 200 episodes (including burn-in)
 - lander_dddqn.py trains in about 500 episodes (including burn-in)
 - lander_policy.py trains in about 1000 episodes, but requires far fewer updates than lander_dddqn.py because only one update is made per episode
 - lander_ac2.py trains in about 1000 episodes, but requires far fewer updates than lander_dddqn.py because only one update is made per episode