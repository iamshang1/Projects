# Reinforcement Learning

This exercise uses the Q-Learning algorithm to choose the best path to reach a target node in a random graph.

The Q-Learning algorithm is a reinforcement learning algorithm that learns the best choice in a given situation through repeated 
trial and error. In this exercise, Q-learning is tasked with finding the best path to a target node from any random node. 
To make the task more complicated, each node on the way to the target may provide a random reward or penalty ranging from 
5 points to -5 points. Reaching the target node provides a reward of 100 points and ends the exercise. 

The Q-Learning algorithm does the following:

1. Set the utility of the target node to 100, and the utility of all other nodes to 0. The utility for a node represents the potential
of recieving a reward (both immediate and long term) by traveling to that node.
2. Start at a random node.
3. Calculate and set the utility of the current node using the formula R + gamma * max(U) where R is the reward of the current node, max(U) 
is the highest utility of any neighbor node, and gamma is a scaling factor between 0 and 1 (we will look at gamma in more detail later)
the utility of every node is set to 0.
4. Move to the neighbor node with the highest utility.
5. Repeat steps 3 and 4 target is reached. As you move through the nodes, the utility of each node is updated to reflect information
from the node's neighbors.
6. Repeat steps 2 through 5 until the utility of every node converges to a stable value.
 
## Results

The initial graph and the immediate reward associated with each node is displayed below. The target has a reward of 100. All other nodes
have a reward between 5 and -5.

![qlearning1](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Reinforcement_Learning/qlearning1.png)

Below is the graph after 500 iterations of Q-learning using a gamma of 0.9. Nodes with higher utility are displayed in lighter colors.
The best path from any node is indicated via a directional arrow (in this plot the arrowheads are the thick ends on each line). We notice
that in general, the best path is usually towards the target.

![qlearning2](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Reinforcement_Learning/qlearning2.png)

Gamma plays an important role in determining the best path to the target. A high gamma like 0.9 means that a high reward will propogate 
through many nodes, which means the potential of long term reward is weighted higher. On the other hand, a low gamma like 0.2 means high
rewards do not propogate very far, so short term reward is more important. Below is the same graph after 500 iterations of Q-learning using
a gamma of 0.2. We notice that the best path in some cases may actually take us farther away from the target or fail to ever reach the target.

![qlearning3](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Reinforcement_Learning/qlearning3.png)

 
## A More Complex Application

Next, we use Q-Learning to create a cat and mouse simulation. The cat is chasing the mice, while the mice are seeking the 
cheese while avoiding the cat. For each step of the simulation, each animal uses the Q-Learning algorithm to generate a new utility matrix
based on its current position relative to obstacles and the other animals. The animal then moves in the direction with the highest utility.
The animation below shows the beginning of the simulation:

![qlearning3](https://github.com/iamshang1/Projects/blob/master/Basic_ML/Reinforcement_Learning/cat_mouse.gif)

It is important to note that Q-Learning does not propogate negative rewards well. This is because Q-Learning will simply avoid the node
with the negative reward, and therefore negative reward values are often ignored in utility values of nearby nodes. To account for this,
the mice in this simulation are trained using two separate utility matrix. The first matrix uses a large positive reward for the cat, and
then inverts all utility values. The second matrix uses a smaller positive reward for the cheese. These two utility matrices are joined to
create the final utility matrix used for mouse movement.