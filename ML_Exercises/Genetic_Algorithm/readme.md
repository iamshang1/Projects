# Simple Genetic Algorithm

Genetic algorithms solve complex problems by generating a large population of possible solutions, testing each solution, selecting the 
solutions with the best performance, and then creating the next generation of solutions using the parameters of the best performers. 
This process is repeated until a satisfactory solution is generated.

In this exercise, a genetic algorithm is used to create an individual that walks from point A to point B in three dimensional space.
Initially, a random population of 1000 individuals is generated. Each individual performs a random walk between 100 and 1000 steps. Each step 
is taken in a random direction of distance <sqrt(3) in three-dimensional space.

The top 50% of individuals who make it closest to the target destination survive to the next generation, while the bottom 50% are eliminated.
Furthermore, the top 25% of individuals randomly exchange their steps to create new individuals that make up 50% of the next generation.

In addition, random step mutations are introduced to 2% of population each generation to prevent stagnation in the evolutionary process. 

## Results

The following images show the path of the individual who makes it closest to the target destination after X generations:

![iteration0](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Genetic_Algorithm/iteration0.png)

![iteration10](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Genetic_Algorithm/iteration10.png)

![iteration20](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Genetic_Algorithm/iteration20.png)

![iteration30](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Genetic_Algorithm/iteration30.png)

![iteration40](https://github.com/iamshang1/Projects/blob/master/ML_Exercises/Genetic_Algorithm/iteration40.png)

We see that it takes between 30 and 40 generations to breed an individual who makes it to the target destination.