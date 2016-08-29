import random
import numpy as np
from scipy.spatial.distance import euclidean
from math import log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#simulation parameters
start = (0,0,0)
target = (100,100,100)
population_size = 1000
iterations = 41

#core functions
def create_individual():
    steps = random.randint(100,1000)
    individual = np.random.rand(steps, 3) * 2 - 1
    return individual

def create_population(size):
    population = [create_individual() for i in range(size)]
    return population

def determine_fitness(individual):
    end = np.sum(individual, axis=0)
    distance = euclidean(end,target)
    fitness = distance * log(len(individual))
    return (fitness,distance)

def determine_population_fitness(population):
    fitness = [determine_fitness(individual)[0] for individual in population]
    pop_fitness = round(float(sum(fitness))/len(population),2)
    return pop_fitness
    
def rank_population(population):
    fitness = [determine_fitness(individual)[0] for individual in population]
    ranking = [i[0] for i in sorted(enumerate(fitness), key=lambda x:x[1])]
    ranked_population = [population[i] for i in ranking]
    return ranked_population

def pairings(reproducing_population):
    shuffle = list(reproducing_population)
    random.shuffle(shuffle)
    pairings = [(shuffle[i],shuffle[i+1]) for i in range(0,len(shuffle)/2,2)]
    return pairings

def reproduce(mother,father):
    mother_genes = range(0,len(mother))
    random.shuffle(mother_genes)
    mother_genes = mother_genes[:len(mother_genes)/2]
    father_genes = range(0,len(father))
    random.shuffle(father_genes)
    father_genes = father_genes[:len(father_genes)/2]
    child = np.concatenate((mother[mother_genes],father[father_genes]))
    return child

def mutate(individual):
    mutation = random.randint(-5,-1)
    mutated_individual = individual[:mutation]
    return mutated_individual

def evolve(ranked_population):
#top 50% live on to next generation, bottom 50% die out
#top 25% randomly reproduce, creating 50% of new population
#random 2% of new population gets genetic mutations
    size = len(ranked_population)
    tophalf = size/2
    topquarter = size/4
    new_population = ranked_population[:tophalf]
    for i in range(8):
        new_population = new_population + [reproduce(mother,father) for mother,father in pairings(ranked_population[:topquarter])]
    for i in range(0, size/50):
        mutation = random.randint(0,size-1)
        new_population[mutation] = mutate(new_population[mutation])
    return new_population

#plot path of fittest individual
def plotfigure(iteration):
    path = np.zeros((len(ranked_population[0]+1),3))
    for i in range(1, len(ranked_population[0]+1)):
        path[i] = path[i-1] + ranked_population[0][i]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = ax.scatter(path[:,0], path[:,1], path[:,2], zdir='z', s=10, c='b', depthshade=True)
    dest = ax.scatter(target[0], target[1], target[2], zdir='z', s=40, c='r', depthshade=True)
    plt.legend([path, dest],["path","target"])
    plt.title("Best Path after %i Iterations" % iteration)
    plt.savefig('iteration%i.png' % iteration)
    plt.show()
    
#run simulation
population = create_population(population_size)
for i in range(0,iterations):
    print "iteration", i+1
    ranked_population = rank_population(population)
    print "(distance from target, steps) for top 5:"
    print [(round(determine_fitness(individual)[1],2), len(individual)) for individual in ranked_population[:5]]
    print "average population fitness (distance*log(steps)):", determine_population_fitness(population)
    print ""
    population = evolve(ranked_population)
    if i%10==0:
        plotfigure(i)