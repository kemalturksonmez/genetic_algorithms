import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
class DE:
    # Creates a network object
    # network - network object
    def __init__(self, network):
        self.net = network
        self.population = [self.net.weights]
        # generate 9 more arrays with similar structures as the weights vector
        for i in range(1, 10):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])

    def binomial_x(self, cross_prob):
        return 0
    
    # trains network using randomized mini batching
    # train_data - training data
    # beta - percentage of vector difference used
    # cross_prob - probability of a cross occuring with target vector and mutation
    # class_outputs - contains a list of class outputs for classification problems
    # num_runs - number of times network is trained
    # batch_size - number of training examples in a batch
    # fitness - average fitness score from inital weight set
    def train(self, train_data, beta, cross_prob, class_outputs, num_runs, batch_size, fitness):
        # copy training data so original training data doesn't get shuffled
        temp_train = deepcopy(train_data)
        for i in range(num_runs):
            # shuffle array with numpy shuffle
            shuffle(temp_train)
            batch = []
            for j in range(batch_size):
                index = int(random.random()*len((temp_train) + 1))
                # print(index)
                batch.append(temp_train[index])
            # update network
            fitness = self.update_network(batch, class_outputs, beta, cross_prob, i % 10, fitness)

    # performs differential evolution on a minibatch
    # mini_batch - mini batch of training data
    # class_outputs - contains a list of class outputs for classification problems
    # beta - percentage of vector difference used
    # cross_prob - probability of a cross occuring with target vector and mutation
    # pop_index - index of current population
    # fitness - average fitness score from previous weight set
    def update_network(self, mini_batch, class_outputs, beta, cross_prob, pop_index, fitness):
        # create trial vector
        u_vect = self.mutate(beta, pop_index)
        # create cross over
        weight_div = self.cross_over(cross_prob, u_vect, pop_index)
        # set network weights as offspring
        self.net.weights = weight_div
        # get average fitness with new weights
        tempFitness, acc = self.net.get_accuracy(mini_batch, class_outputs)
        # print("fitness: ", fitness, " tempFitness: ", tempFitness)
        # if offspring performs better than current candidate solution
        if tempFitness < fitness:
            # update the fitness
            fitness = tempFitness
            # set population at pop_index + 1 to be the new offspring
            self.population[(pop_index + 1) % 10] = weight_div
        # if offspring performed worse
        else:
            # set weights back to the current candidate solution
            self.net.weights = self.population[pop_index]
            # set population at pop_index + 1 as the same 
            self.population[(pop_index + 1) % 10] = self.population[pop_index]
        return fitness
    
    # combines trial vector and current candidate solution to find the next candidate solution
    # cross_prob - probability of a cross occuring with target vector and mutation
    # u_vect - trial vector
    # pop_index - index of current population
    def cross_over(self, cross_prob, u_vect, pop_index):
        weight_div = deepcopy(self.population[pop_index])
        for i in range(len(u_vect)):
            for j in range(len(u_vect[i])):
                for k in range(len(u_vect[i][j])):
                    # if the random value is greater than the crossover probability then 
                    # replace old vector value with new trial value
                    if random.random() > cross_prob:
                        weight_div[i][j][k] = u_vect[i][j][k]
        return weight_div

    # creates trial vector
    def mutate(self, beta, pop_index):
        randVects = [pop_index]
        # find 3 three distinct random indexes
        while len(randVects) != 4:
            randVectIndex = int(random.random()*len((self.population)))
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