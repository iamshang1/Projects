import networkx as nx
import matplotlib.pyplot as plt
from random import randint

# Q-learning parameters
iterations = 500

# Initialize environment
G = nx.random_geometric_graph(50,0.3)
pos = nx.get_node_attributes(G,'pos')
target = randint(0,49)

for node in G.nodes_iter():
    G.node[node]['reward'] = randint(-5,5)
    G.node[node]['utility'] = 0
G.node[target]['utility'] = 100

# Plot starting environment
p = nx.get_node_attributes(G,'reward')
plt.figure(figsize=(10,10))
edges = nx.draw_networkx_edges(G,pos,alpha=0.4)
nodes = nx.draw_networkx_nodes(G,pos,nodelist=p.keys(),node_size=80,node_color=p.values(),cmap=plt.cm.Reds_r)
t = nx.draw_networkx_nodes(G,{target:pos[target]},nodelist=[target],node_size=80,node_color='g')
cbar = plt.colorbar(nodes)
cbar.ax.set_ylabel('reward')
plt.legend([t],['target'])
plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, left=False, right=False)
plt.savefig('qlearning1.png')
plt.show()

# Begin Reinforcement Learning
gamma = 0.9
for i in range(iterations):
    curnode = randint(0,49)
    iter = 0
    while curnode != target and iter < 500:
        try:
            neighbors = G.neighbors(curnode)
            max_utility = -9999999
            for neighbor in neighbors:
                utility = G.node[curnode]['reward'] + gamma*G.node[neighbor]['utility'] 
                if utility > max_utility:
                    max_utility = utility
                    nextnode = neighbor
            G.node[curnode]['utility'] = max_utility
            curnode = nextnode
            iter += 1
        except:
            iter += 1
            curnode = randint(0,49)
      
# Plot final environment
p = nx.get_node_attributes(G,'utility')
F = nx.DiGraph()
for node in G.nodes_iter():
    neighbors = G.neighbors(node)
    next = max(neighbors, key=lambda x:G.node[x]['utility'])
    F.add_edge(node,next)        
plt.figure(figsize=(10,10))
edges = nx.draw_networkx_edges(F,pos,alpha=0.25)
nodes = nx.draw_networkx_nodes(F,pos,nodelist=p.keys(),node_size=80,node_color=p.values(),cmap=plt.cm.Reds_r)
t = nx.draw_networkx_nodes(F,{target:pos[target]},nodelist=[target],node_size=80,node_color='g')
cbar = plt.colorbar(nodes)
plt.legend([t],['target'])
cbar.ax.set_ylabel('utility')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, left=False, right=False)
plt.savefig('qlearning2.png')
plt.show()

# Redo Reinforcement Learning with lower gamma
gamma = 0.2
for i in range(iterations):
    curnode = randint(0,49)
    iter = 0
    while curnode != target and iter < 500:
        try:
            neighbors = G.neighbors(curnode)
            max_utility = -9999999
            nextnode = []
            for neighbor in neighbors:
                utility = G.node[curnode]['reward'] + gamma*G.node[neighbor]['utility'] 
                if utility > max_utility:
                    max_utility = utility
                    nextnode = neighbor
            G.node[curnode]['utility'] = max_utility
            curnode = nextnode
            iter += 1
        except:
            iter += 1
      
# Plot final environment
p = nx.get_node_attributes(G,'utility')
F = nx.DiGraph()
for node in G.nodes_iter():
    neighbors = G.neighbors(node)
    next = max(neighbors, key=lambda x:G.node[x]['utility'])
    F.add_edge(node,next)        
plt.figure(figsize=(10,10))
edges = nx.draw_networkx_edges(F,pos,alpha=0.25)
nodes = nx.draw_networkx_nodes(F,pos,nodelist=p.keys(),node_size=80,node_color=p.values(),cmap=plt.cm.Reds_r)
t = nx.draw_networkx_nodes(F,{target:pos[target]},nodelist=[target],node_size=80,node_color='g')
cbar = plt.colorbar(nodes)
plt.legend([t],['target'])
cbar.ax.set_ylabel('utility')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False, top=False, left=False, right=False)
plt.savefig('qlearning3.png')
plt.show()