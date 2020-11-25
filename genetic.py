# Authors:
# Kemal Turksonmez
# Arash Ajam
import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
import math
''' Performs a Genetic Algorithm in order to tune a neural network '''
class GA:
    # Creates a network object
    # network - network object
    def __init__(self, network):
        self.net = network
        self.popLen = 15
        self.population = [self.net.weights]
        # generate more arrays with similar structures as the weights vector
        for i in range(1, self.popLen):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])

    # evaluates the fitness of the population
    # batch - mini batch of training data
    # class_outputs - list of classification outputs
    def fitness(self, batch, class_outputs):
        fitness = list()
        total=0
        for w in self.population:
            self.net.weights = w
            fit, accuracy = self.net.get_accuracy(batch, class_outputs)
            fitness.append((self.net.weights, fit))
            total += fit
        avg = total/self.popLen
        fitness.sort(key= lambda x:x[1])
        return avg, fitness

    # selections from a sorted list of the population
    # fitness - list of population sorted by fitness
    # rate - percentage of population selected
    # returns:
    # selection - a list of potential parents
    def selection(self, fitness, rate):
        length = self.popLen
        top = int(length * rate)
        # print(top)
        selection = [i[0] for i in fitness[:top]]
        # print(selection)
        return selection
    
    # performs one-point cross over on a list of parents:
    # selection - list of selected parents
    # fit_avg - averate fitness of population
    # mutate_p - mutation probability
    # returns:
    # child_weights - a child after it has been mutated
    def crossover(self, selection, fit_avg, mutate_p):
        length = len(selection)
        rand1 = random.randint(0, len(selection)-1)
        rand2 = random.randint(0, len(selection)-1)
        parent1 = selection[rand1]
        parent2 = selection[rand2]
        # print(rand1, " \n parent1:", parent1, "\n parent2:", parent2)

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
        self.mutate(child_weights, mutate_p)
        # print("mutated \n", child_weights)
        return child_weights


    # performs mutation on a given child
    # weights - weights of a child
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

    # performs GA on a dataset in attempt to find global minimums of the loss function
    # train_data - training date
    # selection_rate - percentage of parents selected from population
    # mutate_p - probability of mutation
    # class_outputs - list of outputs for a classification dataset
    # num_runs - number of generations
    # batch_size - size of a mini batch used for training
    def train(self, train_data, selection_rate, mutate_p, class_outputs, num_runs, batch_size):
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
            # get current population index
            pop_index = i % self.popLen
            # fitness of the group
            avg, fitness = self.fitness(batch, class_outputs)
            # a set of the best parents
            new_select = self.selection(fitness, selection_rate)
            # perform cross over and mutation 
            childs = self.crossover(new_select, avg, mutate_p)
            # get childs fitness and compare it with the worst population member
            self.net.weights = childs
            fit, accuracy = self.net.get_accuracy(batch, class_outputs)
            # remove worst population member
            if fit < fitness[-1][1]:
                 self.population = [i[0] for i in fitness[:self.popLen-1]]
                 self.population.append(childs)
            # else do nothing


