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
        self.swarmLen = 20
        
        self.swarm = [self.net.weights]
        for i in range(1, self.swarmLen):
            self.swarm.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        # set local bests equal to the population
        self.localBest = []
        for i in range(self.swarmLen):
            self.localBest.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        # initialize global best
        self.globalBest = self.net.weights
        
        # ititialize old global best
        self.oldGlobalBest = self.net.weights
        # intialize velocity
        self.velocities = []

        for i in range(self.swarmLen):
            self.velocities.append([np.random.randn(y, x) * 0.1 for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])

        
        
        # for better initialization
        # self.swarm = []
        # self.localBest = []
        # self.globalBest = self.net.weights
        # self.velocities = []
        
    def train(self, train_data, omega, cog_1, cog_2, alpha, class_outputs, num_runs, batch_size):
        globalFitness, acc = self.net.get_accuracy(train_data, class_outputs)
        # initialize global best to be the global best
        for i in range(self.swarmLen):
            # set weights
            self.net.weights = self.swarm[i]
            fitness, acc = self.net.get_accuracy(train_data, class_outputs)
            if fitness < globalFitness:
                self.oldGlobalBest = self.globalBest
                self.globalBest = self.swarm[i]
                globalFitness = fitness
            
        # for i in range(self.swarmLen):
        #     velocity = [np.random.rand(y, x) * 0.1 for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])]
        #     for layer in range(len(velocity)):
        #         velocity[layer] = velocity[layer] * v_max
        #     self.velocities.append(velocity)
        
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
            
            # print(pop_index, self.swarm[pop_index][0][0][1], self.globalBest[0][0][1])
            
            # once every particle has been updated in a generation, update velocity then weights
            if pop_index == 0 and i > 0:
                for particleIndex,particle in enumerate(self.swarm):
                    for layer in range(len(particle)):
                        # print(self.velocities[particleIndex][layer][0])
                        # print(self.velocities[particleIndex][layer][0])
                        self.velocities[particleIndex][layer] = (omega * self.velocities[particleIndex][layer]) 
                        + (cog_1 * random.random() * (self.localBest[particleIndex][layer] - particle[layer])) 
                        + (cog_2 * random.random() * (self.globalBest[layer] - particle[layer]))
                        # print(self.velocities[particleIndex][layer][0])
                        # print()
                        
                    # # keep velocity in check
                    #     for wv in range(len(self.velocities[particleIndex][layer])):
                    #         for c in  range(len(self.velocities[particleIndex][layer][wv])):    
                    #             if self.velocities[particleIndex][layer][wv][c] > v_max:
                    #                 self.velocities[particleIndex][layer][wv][c] = v_max
                    #             elif self.velocities[particleIndex][layer][wv][c] < -v_max:
                    #                 self.velocities[particleIndex][layer][wv][c] = -v_max
                

                # update positions
                for particleIndex in range(self.swarmLen):
                    for layer in range(len(self.swarm[particleIndex])):
                        self.swarm[particleIndex][layer] = self.swarm[particleIndex][layer] + self.velocities[particleIndex][layer]
                # update omega using linear intertia reduction
                omega = omega * alpha
                
            # compare global and local bests to current position
            self.update_network(batch, class_outputs, pop_index)

        # set weights as global best
        self.net.weights = self.globalBest
        globalFitness, acc = self.net.get_accuracy(train_data, class_outputs)
        # print("GLOBAL FITNESS", globalFitness)
        # find global best to be the global best
        for i in range(self.swarmLen):
            # set weights
            self.net.weights = self.swarm[i]
            fitness, acc = self.net.get_accuracy(train_data, class_outputs)
            # print("FITNESS", fitness)
            if fitness < globalFitness:
                self.globalBest = self.swarm[i]
                globalFitness = fitness
        # set weights as global best
        self.net.weights = self.globalBest
        
    
    def update_network(self, batch, class_outputs, pop_index):
        
        # get best local fitness level of particle
        self.net.weights = self.localBest[pop_index]
        localBestFitness, acc = self.net.get_accuracy(batch, class_outputs)
        # if pop_index == 7:
        #     print("LOCAL FITNESS", localBestFitness)
        # get current fitness level of particle
        self.net.weights = self.swarm[pop_index]
        fitness, acc = self.net.get_accuracy(batch, class_outputs)
        # if pop_index == 7:
        #     print(pop_index, " FITNESS", fitness)
        # set the personal best position
        if fitness < localBestFitness:
            localBestFitness = fitness
            self.localBest[pop_index] = self.swarm[pop_index]

        # get best fitness level of swarm
        self.net.weights = self.globalBest
        globalBestFitness, acc = self.net.get_accuracy(batch, class_outputs)
        # if pop_index == 7:
        #     print("Global FITNESS", globalBestFitness)
        # set the neighborhood best position
        if fitness < globalBestFitness:
            self.oldGlobalBest = self.globalBest
            self.globalBest = self.swarm[pop_index]


