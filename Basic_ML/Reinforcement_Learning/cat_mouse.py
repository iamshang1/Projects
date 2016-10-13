import numpy as np
from copy import copy
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

# Parameters
world_size = 10
gamma = 0.5
q_learning_iterations = 250
cats = 1
mice = 2
cheese_rate = 0.01
obstacle_rate = 0.3
sight_range = 8
cheese_value = 50
cat_value = -100

# Classes
class Cat(object):
    def __init__(self, position, sight_range=4, gamma=0.8, q_learning_iterations=100):
        self.position = position
        self.sight_range = sight_range
        self.gamma = gamma
        self.q_learning_iterations = q_learning_iterations
    def update(self,world):
        self.world = world
        self.vision = self.world[self.position[0]-self.sight_range:self.position[0]+self.sight_range+1,self.position[1]-self.sight_range:self.position[1]+self.sight_range+1]
    def q_learn(self):
        shape = self.vision.shape
        self.utility_grid = np.zeros((shape[0],shape[0]))
        for i in range(0,shape[0]):
            for j in range(0,shape[0]):
                if self.vision[i,j] == 2:
                    self.utility_grid[i,j] = 100
        for i in range(self.q_learning_iterations):
            start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            while self.vision[start[0],start[1]] == 9 or self.vision[start[0],start[1]] == 1 or self.vision[start[0],start[1]] == 3:
                start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            pos = start
            for i in range(100):
                if self.vision[pos[0],pos[1]] == 2:
                    break
                neighbors = [(pos[0]+1,pos[1]),(pos[0]-1,pos[1]),(pos[0],pos[1]+1),(pos[0],pos[1]-1)]
                max_utility = -999
                next = pos
                for neighbor in neighbors:
                    if neighbor[0] > shape[0]-1 or neighbor[1] > shape[0]-1 or neighbor[0] < 0 or neighbor[1] < 0:
                        continue
                    elif self.vision[neighbor[0],neighbor[1]] == 9 or self.vision[neighbor[0],neighbor[1]] == 1 or self.vision[neighbor[0],neighbor[1]] == 3:
                        continue
                    utility = round(self.reward_grid[pos[0],pos[1]] + self.gamma*self.utility_grid[neighbor[0],neighbor[1]], 2)
                    if utility > max_utility:
                        max_utility = utility
                        next = neighbor
                self.utility_grid[pos[0],pos[1]] = max_utility
                pos = next
    def move(self):
        self.reward_grid = np.ones((self.sight_range*2+1,self.sight_range*2+1)) * -0.1
        self.q_learn()
        options = {"up":(self.sight_range-1,self.sight_range),"down":(self.sight_range+1,self.sight_range),"left":(self.sight_range,self.sight_range-1),"right":(self.sight_range,self.sight_range+1)}
        self.nextmove = "stay"
        max_utility = -999
        for key,val in options.iteritems():
            if self.vision[val[0],val[1]] == 9 or self.vision[val[0],val[1]] == 1 or self.vision[val[0],val[1]] == 3:
                continue
            utility = self.utility_grid[val[0],val[1]]
            if utility > max_utility:
                max_utility = utility
                self.nextmove = key
        current = copy(self.position)
        if self.nextmove == "up":
            self.position[0] -= 1
            self.world[self.position[0],self.position[1]] = 1
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "down":
            self.position[0] += 1
            self.world[self.position[0],self.position[1]] = 1
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "left":
            self.position[1] -= 1
            self.world[self.position[0],self.position[1]] = 1
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "right":
            self.position[1] += 1
            self.world[self.position[0],self.position[1]] = 1
            self.world[current[0],current[1]] = 0
        return self.world

class Mouse(object):
    def __init__(self, position, sight_range=4, gamma=0.8, q_learning_iterations=100, cheese_value = 10, cat_value = -100):
        self.position = position
        self.sight_range = sight_range
        self.gamma = gamma
        self.q_learning_iterations = q_learning_iterations
        self.cheese_value = cheese_value
        self.cat_value = cat_value
    def update(self,world):
        self.world = world
        self.vision = self.world[self.position[0]-self.sight_range:self.position[0]+self.sight_range+1,self.position[1]-self.sight_range:self.position[1]+self.sight_range+1]
    def q_learn(self):
        shape = self.vision.shape
        self.utility_grid_cat = np.zeros((shape[0],shape[0]))
        self.utility_grid_cheese = np.zeros((shape[0],shape[0]))
        for i in range(0,shape[0]):
            for j in range(0,shape[0]):
                if self.vision[i,j] == 1:
                    self.utility_grid_cat[i,j] = -self.cat_value
                if self.vision[i,j] == 3:
                    self.utility_grid_cheese[i,j] = self.cheese_value
        for i in range(self.q_learning_iterations):
            start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            while self.vision[start[0],start[1]] == 9 or self.vision[start[0],start[1]] == 2:
                start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            pos = start
            for i in range(100):
                if self.vision[pos[0],pos[1]] == 1:
                    break
                neighbors = [(pos[0]+1,pos[1]),(pos[0]-1,pos[1]),(pos[0],pos[1]+1),(pos[0],pos[1]-1)]
                max_utility = -999
                next = pos
                for neighbor in neighbors:
                    if neighbor[0] > shape[0]-1 or neighbor[1] > shape[0]-1 or neighbor[0] < 0 or neighbor[1] < 0:
                        continue
                    elif self.vision[neighbor[0],neighbor[1]] == 9 or self.vision[neighbor[0],neighbor[1]] == 2:
                        continue
                    utility = round(self.reward_grid[pos[0],pos[1]] + self.gamma*self.utility_grid_cat[neighbor[0],neighbor[1]],2)
                    if utility > max_utility:
                        max_utility = utility
                        next = neighbor
                self.utility_grid_cat[pos[0],pos[1]] = max_utility
                pos = next
        for i in range(self.q_learning_iterations):
            start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            while self.vision[start[0],start[1]] == 9 or self.vision[start[0],start[1]] == 2:
                start = (np.random.randint(shape[0]),np.random.randint(shape[0]))
            pos = start
            for i in range(100):
                if self.vision[pos[0],pos[1]] == 3:
                    break
                neighbors = [(pos[0]+1,pos[1]),(pos[0]-1,pos[1]),(pos[0],pos[1]+1),(pos[0],pos[1]-1)]
                max_utility = -999
                next = pos
                for neighbor in neighbors:
                    if neighbor[0] > shape[0]-1 or neighbor[1] > shape[0]-1 or neighbor[0] < 0 or neighbor[1] < 0:
                        continue
                    elif self.vision[neighbor[0],neighbor[1]] == 9 or self.vision[neighbor[0],neighbor[1]] == 2:
                        continue
                    utility = round(self.reward_grid[pos[0],pos[1]] + self.gamma*self.utility_grid_cheese[neighbor[0],neighbor[1]],2)
                    if utility > max_utility:
                        max_utility = utility
                        next = neighbor
                self.utility_grid_cheese[pos[0],pos[1]] = max_utility
                pos = next
        self.utility_grid = self.utility_grid_cheese - self.utility_grid_cat
    def move(self):
        self.reward_grid = np.ones((self.sight_range*2+1,self.sight_range*2+1)) * 0.1
        self.q_learn()
        options = {"up":(self.sight_range-1,self.sight_range),"down":(self.sight_range+1,self.sight_range),"left":(self.sight_range,self.sight_range-1),"right":(self.sight_range,self.sight_range+1)}
        self.nextmove = "stay"
        max_utility = -999
        for key,val in options.iteritems():
            if self.vision[val[0],val[1]] == 9 or self.vision[val[0],val[1]] == 2:
                continue
            utility = self.utility_grid[val[0],val[1]]
            if utility > max_utility:
                max_utility = utility
                self.nextmove = key
        current = copy(self.position)
        if self.nextmove == "up":
            self.position[0] -= 1
            self.world[self.position[0],self.position[1]] = 2
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "down":
            self.position[0] += 1
            self.world[self.position[0],self.position[1]] = 2
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "left":
            self.position[1] -= 1
            self.world[self.position[0],self.position[1]] = 2
            self.world[current[0],current[1]] = 0
        elif self.nextmove == "right":
            self.position[1] += 1
            self.world[self.position[0],self.position[1]] = 2
            self.world[current[0],current[1]] = 0
        return self.world

class World(object):
    def __init__(self,world_size=100, gamma=0.8, q_learning_iterations=100, cats=2, mice=5, cheese_rate=0.001, obstacle_rate=0.01, sight_range=4, cheese_value = 10, cat_value = -100):
        self.world = np.zeros((world_size+sight_range*2,world_size+sight_range*2))
        self.world[:,:sight_range] = 9
        self.world[:,-sight_range:] = 9
        self.world[:sight_range,:] = 9
        self.world[-sight_range:,:] = 9
        self.sight_range = sight_range
        self.gamma = gamma
        self.q_learning_iterations = q_learning_iterations
        self.num_cats = cats
        self.num_mice = mice
        self.cheese_rate = cheese_rate
        self.obstacle_rate = obstacle_rate
        self.cheese_value = cheese_value
        self.cat_value = cat_value
        self.cats = []
        self.mice = []
        #self.gif = 0
        self.populate()
        self.state()
    def populate(self):
        taken = []
        for cat in range(self.num_cats):
            position = [np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range),np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range)]
            while position in taken:
                position = [np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range),np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range)]
            taken.append(position)
            self.cats.append(Cat(position,self.sight_range,self.gamma,self.q_learning_iterations))
            self.world[position[0],position[1]] = 1
        for mouse in range(self.num_mice):
            position = [np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range),np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range)]
            while position in taken:
                position = [np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range),np.random.randint(self.sight_range,self.world.shape[0]-self.sight_range)]
            taken.append(position)
            self.mice.append(Mouse(position,self.sight_range,self.gamma,self.q_learning_iterations,self.cheese_value,self.cat_value))
            self.world[position[0],position[1]] = 2
        for i in range(0,self.world.shape[0]):
            for j in range(0,self.world.shape[0]):
                if self.world[i,j] == 0:
                    if np.random.random() < self.cheese_rate:
                        self.world[i,j] = 3
        for i in range(0,self.world.shape[0]):
            for j in range(0,self.world.shape[0]):
                if self.world[i,j] == 0:
                    if np.random.random() < self.obstacle_rate:
                        self.world[i,j] = 9
        for cat in self.cats:
            cat.update(self.world)
        for mouse in self.mice:
            mouse.update(self.world)
    def iterate(self):
        for mouse in self.mice:
            if self.world[mouse.position[0],mouse.position[1]] == 1:
                self.mice.remove(mouse)
                continue
            self.world = mouse.move()
            for cat in self.cats:
                cat.update(self.world)
            for mouse in self.mice:
                mouse.update(self.world)
            self.state()
        for cat in self.cats:
            self.world = cat.move()
            for cat in self.cats:
                cat.update(self.world)
            for mouse in self.mice:
                mouse.update(self.world)
            self.state()
        for i in range(0,self.world.shape[0]):
            for j in range(0,self.world.shape[0]):
                if self.world[i,j] == 0:
                    if np.random.random() < self.cheese_rate:
                        self.world[i,j] = 3
        for cat in self.cats:
            cat.update(self.world)
        for mouse in self.mice:
            mouse.update(self.world)
        self.state()
    def state(self):
        img = np.copy(self.world[self.sight_range:self.world.shape[0]-self.sight_range,self.sight_range:self.world.shape[0]-self.sight_range])
        img[img==0] = 256
        img[img==1] = 215
        img[img==2] = 123
        img[img==3] = 175
        img[img==9] = 1
        p = plt.imshow(img, interpolation='nearest', cmap='nipy_spectral')
        fig = plt.gcf()
        c1 = mpatches.Patch(color='red', label='cats')
        c2 = mpatches.Patch(color='green', label='mice')
        c3 = mpatches.Patch(color='yellow', label='cheese')
        plt.legend(handles=[c1,c2,c3],loc='center left',bbox_to_anchor=(1, 0.5))
        #plt.savefig("cat_mouse%i.png" % self.gif, bbox_inches='tight')
        #self.gif += 1
        plt.pause(0.1)
        
# Run algorithm
World = World(world_size,gamma,q_learning_iterations,cats,mice,cheese_rate,obstacle_rate,sight_range,cheese_value,cat_value)
for i in range(40):
    print "iteration %i" % (i+1)
    World.iterate()