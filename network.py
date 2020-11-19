# Authors:
# Kemal Turksonmez
# Arash Ajam
import numpy as np
from math import exp
from math import log
import matplotlib.pyplot as plt
# This class contains methods to build, train, and test an MLP
class Network():
    # Constructs a network object and creates an array to define the properties of the network
    # n_hidden - number of hidden layers
    # n_outputs - number of outputs
    # n_inputs - number of inputs
    # layer_nodes - number of nodes in a hidden layer
    # problemType - identifies if its regression or classification
    def __init__(self, n_hidden, n_outputs, n_inputs, layer_nodes, problemType):
        self.n_outputs = n_outputs
        self.net_props = [n_inputs] 
        self.n_hidden = n_hidden
        self.layer_nodes = layer_nodes
        self.net_props.extend([layer_nodes] * n_hidden)
        self.net_props.append(n_outputs)
        self.problemType = problemType
        print("Number of inputs: ", n_inputs)
        print("Number of hidden layer nodes: ", layer_nodes)
        print("Number of hidden layers: ", n_hidden)
        print("Number of outputs: ", n_outputs)
        print("Network Properties: ", self.net_props)
        self.initialize_network(self.net_props)
        

    # Initializes the weights and biases based on the network properities
    # net_props - properties of the network (defines the number of nodes)
    def initialize_network(self,net_props):
            # intialize random biases
            self.biases = [np.random.randn(y, 1) for y in self.net_props[1:]]
            if self.problemType == "regression":
                self.biases = [np.zeros((y,1)) for y in self.net_props[1:]]
            # intialize random weights
            self.weights = [np.random.randn(y, x) for x, y in zip(self.net_props[:-1], self.net_props[1:])]

    # performs feed forward calculation and stores calculated activations layers
    # inp - input values
    # returns:
    # out - output of final layer
    # activations - activation values from each layer
    def feedforward(self, inp):
        # stores all the activations layer by layer
        activations = [inp] 
        # tracks index 
        index = 0
        out = inp
        for b,w in zip(self.biases, self.weights): 
            # print("Layer ", index+2)
            # print("Weights: ", w)
            # print("Inputs: ", out)
            # print("Bias: ", b)
            # print("Z: ", np.dot(w, out)+b)
            # check to see if its the output layer
            if index == (len(self.weights)-1):
                # check problem type
                if self.problemType == 'regression':
                    # linear activation
                    out = np.dot(w, out)+b
                else:
                    # softmax
                    out = self.softmax(np.dot(w, out)+b)                  
            else:
                out = self.sigmoid(np.dot(w, out)+b)    
            activations.append(out)
            index += 1
            # print("Output: ", out)
        return out, activations
        
    
    # creates an array of expected outputs for a given example
    # expected - class of expected output
    # class_outputs - list of classes in the dataset
    # returns
    # array of expected outputs
    def one_hot_encoding(self, expected, class_outputs):
        temp = []
        for key in class_outputs:
            # winning output should have a value of 1
            if key == expected[0]:
                temp.append([1])
            # every other output should have a value of 0
            else:
                temp.append([0])
        return temp

    


    # performs sigmoid function on a value
    # z - value
    # returns:
    # the result of the sigmoid function
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    # derivative of sigmoid function
    # z - a value that has already had the sigmoid applied to it
    # returns:
    # the result of the sigmoid derivative function
    def sigmoid_drvt(self, z):
        return z*(1-z)
    
    # cross entroy of outputs
    # output - one hot encoding
    # y - output vector that has had the softmax function applied to it
    # returns 
    # cross entropy
    def cross_entropy(self, output, y):
        y = np.log(1e-15 + y)
        sumAll = 0
        for i in range(len(output)):
            sumAll += output[i] * y[i]
        return -sumAll[0]

    # Mean Squared Error of Output
    # output - output of network
    # y - expected output
    # returns 
    # mean squared error
    def mse(self, output, y):
        return (output[0] - y[0])**2

    # calculates the derivative of the cross entropy loss between expected and actual output
    # output - actual output
    # y - expected output
    # returns:
    # derivative of the output
    def cross_entropy_drvt(self, output, y):
        return (output-y)
    
    # caluclates softmax of outputs
    # modified to be numerically stable
    # output - output vector of the neural network
    def softmax(self, output):
        exps = np.exp(output - np.max(output))
        return exps / np.sum(exps)
    
    # graph loss functions of best performing training set
    # filename - name of file
    # train_data - training data
    # test_data - test data
    # class_outputs - list of classes in the dataset
    # batch_size - number of items in the batch set
    # num_rums - number of iterations the network will train
    # lr - learning rate of network
    # momentum - momentum of gradient calculation
    def graph(self, filename, train_data, test_data, class_outputs, batch_size, num_runs, lr, momentum=0):
        size = 4
        numPoints = int(num_runs/size)
        lossArr = []
        for i in range(numPoints):
            loss = 0
            # run stochaistic 
            self.stochastic_GD(train_data, class_outputs, batch_size, size, lr, momentum=0)
            for row in test_data:
                out, a = self.feedforward(row[:-1])
                one_hot = self.one_hot_encoding(row[-1], class_outputs)
                if self.problemType == "classification":
                    loss += self.cross_entropy(one_hot, out)
                else:
                    loss += self.mse(row[-1], out[0])
            lossArr.append(loss/len(test_data))

        error_name = "Cross Entropy Loss"
        if self.problemType == "regression":
            error_name = " Mean Squared Error"
        plt.plot(lossArr)
        plt.xlabel("Training Iteration [Multiplied by " + str(size) + "]")
        plt.ylabel(error_name)
        plt.title('Error vs Training Iteration on ' + str(filename) + " With " + str(self.n_hidden) + " Layers", fontsize=12)
        plt.savefig('results/' + filename + '_' + str(self.n_hidden) +'_layers_loss.png', dpi=600, bbox_inches='tight')
        plt.clf()

    # passes each test example to the neural network and measures each output with respective method
    # test_data - set of test data
    # class_outputs - one hot encoding of expected values (empty for regression)
    # returns:
    # loss - value of loss function applied to network
    # accuracy - returns percentage of guesses that were correct (classification only)
    def get_accuracy(self, test_data, class_outputs):
        count = 0
        loss = 0
        if self.problemType == "classification":
            
            for row in test_data:
                one_hot = self.one_hot_encoding(row[-1], class_outputs)
                out, activations = self.feedforward(row[:-1])
                bestScore, bestIndex = 0,0
                loss += self.cross_entropy(one_hot, out)
                for i in range(len(out)):
                    if bestScore < out[i]:
                        bestScore = out[i]
                        bestIndex = i
                # if it's correct it adds a one, if its wrong it adds a zero
                count += one_hot[bestIndex][0]
        else:
            # MSE
            for row in test_data:
                out, activations = self.feedforward(row[:-1])
                # loss += (row[-1][0] - out[0][0])**2
                loss += self.mse(out[0], row[-1])
        return loss/len(test_data), count/len(test_data)
