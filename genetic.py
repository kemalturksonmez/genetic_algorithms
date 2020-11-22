import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
class GA:
    # Creates a network object
    # network - network object
    def __init__(self, network):
        self.net = network
        self.population = [self.net.weights]
        # generate 9 more arrays with similar structures as the weights vector
        for i in range(1, 10):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        print(self.population[0])
        self.crossover(self.population)
    # def create_population(pop_n):
        # for n in pop_n:
            # create network


    def fitness(networks):
        # fitness = dict()
        # for net in networks:
        #     fitness.add(net, net.accuracy)
        # fitness.sort()
        # return fitness

    def selection(self, fitness, networks, rate):
        # length = len(networks)
        # selection = fitness.getTopRate()
        # return selection
        
    def crossover(self, selection):
        length = len(selection)
        rand1 = random.randint(0, len(selection)-1)
        rand2 = random.randint(0, len(selection)-1)
        parent1 = selection[rand1]
        parent2 = selection[rand2]
        print(rand1, " \n parent1:", parent1, "\n parent2:", parent2)

        w1 = np.array(parent1)
        w2 = np.array(parent2)
        
        # properties = [parent1.input_nodes, parent1.output_nodes, parent1.hidden_nodes, parent1.hidden_layers]
        
        # initializes the childs to zeros
        # child = [np.zeros(w1.shape) for w in self.net.weights]
        # child2 = [np.zeros(w1.shape) for w in self.net.weights]

        child_weights = []

        # Iterate through parent1 weights
        for i in range(0, len(parent1)):
            
            # Get single point to split the matrix in parents based on # of cols
            # split = random.randint(0, np.shape(parent1[i])[1]-1)
            split =  int(len(parent1[i])/2)
            # print("split\n", split)
            
            # Iterate through elements after the split point and swap columns after it for parent1 and parent2 
            for j in range(split, np.shape(parent1[i])[1]):
                parent1[i][:, j] = parent2[i][:, j]

            # After crossover add weights to child
            child_weights.append(parent1[i])
        # print("child \n", child_weights)
        self.mutate(child_weights, .2)
        # print("mutated \n", child_weights)
        return child_weights


    # weights: 
    # mutate_p - probability of mutation
    def mutate(self, weights, mutate_p):
        # random value to be added to an element
        rand_value = random.uniform(-1,1)
        # print(rand_value)
        for i, layer in enumerate(weights):
            for j, row in enumerate(layer):
                for k, w in enumerate(row):
                    rand_p = random.random()
                    # print(rand_p, k, j, i)
                    # print(w + rand_value)
                    if(rand_p < mutate_p):
                        w = w + rand_value
                        weights[i][j][k] = w
