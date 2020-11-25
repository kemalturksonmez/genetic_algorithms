# Authors:
# Kemal Turksonmez
# Arash Ajam
import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
import math
''' Performs Differential Evolution in order to tune a neural network '''
class DE:
    # Creates a network object
    # network - network object
    def __init__(self, network):
        self.net = network
        self.population = [self.net.weights]
        self.popLen = 15
        # generate 9 more arrays with similar structures as the weights vector
        for i in range(1, self.popLen):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        self.nextGen = self.population

    
    # trains network using DE/curr/1/bin
    # train_data - training data
    # beta - percentage of vector difference used
    # cross_prob - probability of a cross occuring with target vector and mutation
    # class_outputs - contains a list of class outputs for classification problems
    # num_runs - number of times network is trained
    # batch_size - number of training examples in a batch
    # fitness - average fitness score from inital weight set
    def train(self, train_data, beta, cross_prob, class_outputs, num_runs, batch_size):
        # copy training data so original training data doesn't get shuffled
        temp_train = deepcopy(train_data)
        # used to decide what stretch of training data should be taken from the shuffled data set
        numOcc = math.ceil(len(train_data)/batch_size)
        for i in range(num_runs):
            index = (i % numOcc)
            if  index == 0:
                shuffle(temp_train)
            # shuffle array with numpy shuffle
            batch = temp_train[index * batch_size : (index * batch_size) + batch_size]
            # set new generation
            if (i % self.popLen) == 0:
                self.population = self.nextGen
            # curr index
            pop_index = i % self.popLen
            
            fitness = self.update_network(batch, class_outputs, beta, cross_prob, pop_index)
        # finds the most fit member of the population
        bestFitness = float('inf')
        # pop index
        bestIndex = 0
        for index, weights in enumerate(self.population):
            self.net.weights = weights
            # get fitness
            fitness, acc = self.net.get_accuracy(train_data, class_outputs)
            if fitness < bestFitness:
                bestFitness = fitness
                bestIndex = index
        self.net.weights = self.population[bestIndex]

            

    # performs differential evolution on a minibatch
    # mini_batch - mini batch of training data
    # class_outputs - contains a list of class outputs for classification problems
    # beta - percentage of vector difference used
    # cross_prob - probability of a cross occuring with target vector and mutation
    # pop_index - index of current population
    # fitness - average fitness score from previous weight set
    # returns:
    # fitness - fitness of current population member
    def update_network(self, mini_batch, class_outputs, beta, cross_prob, pop_index):
        # # set network weights as pop_index
        self.net.weights = self.population[pop_index]
        # # get fitness
        fitness, acc = self.net.get_accuracy(mini_batch, class_outputs)
        # create trial vector
        ###########
        temp = self.net.weights
        ############
        u_vect = self.mutate(beta, pop_index)
        # create cross over
        weight_div = self.cross_over(cross_prob, u_vect, pop_index)
        # set network weights as offspring
        self.net.weights = weight_div
        # get average fitness with new weights
        tempFitness, acc = self.net.get_accuracy(mini_batch, class_outputs)

        # if offspring performs better than current candidate solution
        if tempFitness < fitness:
            # update the fitness
            fitness = tempFitness
            # set the next generation population to be the new offspring
            self.nextGen[pop_index] = weight_div
        # if offspring performed worse
        else:
            # self.net.weights = self.population[pop_index]
            self.net.weights = temp
            # set the next generation population to be the same 
            self.nextGen[pop_index] = temp
        return fitness
    
    # combines trial vector and current candidate solution to find the next candidate solution
    # cross_prob - probability of a cross occuring with target vector and mutation
    # u_vect - trial vector
    # pop_index - index of current populations
    # returns:
    # weight_div - mutated trial vector
    def cross_over(self, cross_prob, u_vect, pop_index):
        weight_div = deepcopy(self.population[pop_index])
        for i in range(len(u_vect)):
            for j in range(len(u_vect[i])):
                for k in range(len(u_vect[i][j])):
                    # if the random value is greater than the crossover probability then 
                    # replace old vector value with new trial value
                    randVal = random.random()
                    if randVal > cross_prob:
                        weight_div[i][j][k] = u_vect[i][j][k]
        return weight_div

    # creates trial vector
    # beta - mulitiplier of trial vector
    # pop_index - current observed population member
    # returns:
    # u_vect - trial vector
    def mutate(self, beta, pop_index):
        randVects = [pop_index, pop_index]

        # find 3 three distinct random indexes
        while len(randVects) != 4:
            randVectIndex = int(random.random()*self.popLen)
            if not randVectIndex in randVects:
                randVects.append(randVectIndex)
        # copy without reference
        u_vect = deepcopy(self.population[randVects[2]])

        # subtract x_3 from x_2
        for i in range(len(u_vect)):
            u_vect[i] = self.population[randVects[2]][i] - self.population[randVects[3]][i]

        # muliply by beta
        for i in range(len(u_vect)):
            u_vect[i] = u_vect[i] * beta

        # add u_vect to x_1
        for i in range(len(u_vect)):
            u_vect[i] = self.population[randVects[1]][i] + u_vect[i]

        return u_vect