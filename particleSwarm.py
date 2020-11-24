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
        self.popLen = 10
        self.population = [self.net.weights]
        for i in range(1, self.popLen):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        # set local bests equal to the population
        self.localBest = self.population
        # get random global best
        self.globalBest = self.populationint(random.random()*self.popLen))
        
        
    def train(self, train_data, omega, cog_1, cog_2, class_outputs, num_runs, batch_size):
        return 0