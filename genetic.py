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
        for i in range(1, 30):
            self.population.append([np.random.randn(y, x) for x, y in zip(self.net.net_props[:-1], self.net.net_props[1:])])
        # print(self.population[0])
        # self.crossover(self.population)
    # def create_population(pop_n):
        # for n in pop_n:
            # create network


    def fitness(self, batch, class_outputs):
        fitness = list()
        total=0
        for w in self.population:
            self.net.weights = w
            fit, accuracy = self.net.get_accuracy(batch, class_outputs)
            fitness.append((self.net.weights, fit))
            total += fit
        avg = total/len(self.population)
        fitness.sort(key= lambda x:x[1], reverse=True)
        return avg, fitness

    def selection(self, fitness, rate):
        length = len(self.population)
        top = int(length * rate)
        # print(top)
        selection = [i[0] for i in fitness[:top]]
        # print(selection)
        return selection
        
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

    def generate(self, batch, class_outputs, selection_rate, mutate_p):
        count = 0
        convergence = False
        while(not convergence):
            count+=1
            print("Genration ", count)
            avg, fitness = self.fitness(batch, class_outputs)
            new_select = self.selection(fitness, selection_rate)
            childs = self.crossover(new_select, avg, mutate_p)
            self.population.append(childs)
            avg, fitness = self.fitness(batch, class_outputs)
            print('avg', avg)
            if(fitness[0][1] > 9 or count> 1000):
                convergence = True
            print("fittest", fitness[0][1])
            print("second fittest", fitness[1][1])
        fitness, avg = self.fitness(batch, class_outputs)
        # print(fitness, avg)

    def train(self, train_data, selection_rate, mutate_p, class_outputs, batch_size):
        # copy training data so original training data doesn't get shuffled
        temp_train = deepcopy(train_data)
        # shuffle array with numpy shuffle
        shuffle(temp_train)
        batch = []
        for j in range(batch_size):
            index = int(random.random()*len((temp_train) + 1))
            # print(index)
            batch.append(temp_train[index])
        # update network
        self.generate(batch, class_outputs, selection_rate, mutate_p)
