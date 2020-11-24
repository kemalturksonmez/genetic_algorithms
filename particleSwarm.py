# Authors:
# Kemal Turksonmez
# Arash Ajam
import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
import math

class PSO:
    # Creates a network object
    # network - network object
    def __init__(self, network):
        self.net = network
        self.swarmLen = 10
        self.swarm = [self.net.weights]
        for i in range(1, self.swarmLen):
            self.swarm.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        # set local bests equal to the population
        self.localBest = self.swarm
        # get random global best
        self.globalBest = self.swarm[int(random.random()*self.swarmLen)]
        # intialize velocity
        self.velocities = []
        for i in range(self.swarmLen):
            self.velocities.append([np.random.randn(y, x)  * 0.1 for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        
        
    def train(self, train_data, omega, cog_1, cog_2, class_outputs, num_runs, batch_size):
        temp_train = deepcopy(train_data)
        # used to decide what stretch of training data should be taken from the shuffled data set
        numOcc = math.ceil(len(train_data)/batch_size)
        for i in range(num_runs):
            index = (i % numOcc)
            # if entire dataset has been iterated through, shuffle data and start from the beginning
            if index == 0:
                shuffle(temp_train)
            # shuffle array with numpy shuffle
            batch = temp_train[index * batch_size : (index * batch_size) + batch_size]

            pop_index = i % self.swarmLen
            
            
            # once every particle has been updated in a generation, update velocity then weights
            if pop_index == 0:
                # updates weights
                for particleIndex,particle in enumerate(self.swarm):
                    self.velocities[particleIndex] = [(omega * self.velocities[particleIndex][0]) 
                    + (cog_1 * random.random() * (self.localBest[particleIndex][0] - particle[0])) 
                    + (cog_2 * random.random() * (self.localBest[particleIndex][0] - particle[0]))]
                # update positions
                for particleIndex in range(self.swarmLen):
                    self.swarm[particleIndex] = [self.swarm[particleIndex][0] + self.velocities[particleIndex][0]]

            # compare global and local bests to current position
            self.update_network(batch, class_outputs, pop_index)

        # set weights as global best
        self.net.weights = self.globalBest
    
    def update_network(self, batch, class_outputs, pop_index):
        # get best local fitness level of particle
        self.net.weights = self.localBest[pop_index]
        localBestFitness, acc = self.net.get_accuracy(batch, class_outputs)
        # get current fitness level of particle
        self.net.weights = self.swarm[pop_index]
        fitness, acc = self.net.get_accuracy(batch, class_outputs)
        # set the personal best position
        if fitness < localBestFitness:
            self.localBest[pop_index] = self.swarm[pop_index]
        # get best fitness level of swarm
        self.net.weights = self.globalBest
        globalBestFitness, acc = self.net.get_accuracy(batch, class_outputs)
        # set the neighborhood best position
        if fitness < globalBestFitness:
            self.globalBest = self.swarm[pop_index]

