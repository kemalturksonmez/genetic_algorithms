# Authors:
# Kemal Turksonmez
# Arash Ajam
import random
import numpy as np
from copy import deepcopy
from numpy.random import shuffle
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
    
    # performs stochiastic gradient descent
    # train_data - training data set
    # class_outputs - contains a list of class outputs for classification problems
    # batch_size - number of test rows in a mini batch
    # num_runs - number of training iterations
    # lr - learning rate of back propogation
    # momentum - momentum of back propogation
    def stochastic_GD(self, train_data, class_outputs, batch_size, num_runs, lr, momentum=0):
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
                # batch.append(temp_train[j])
            # update network
            self.update_network(batch, class_outputs, lr, momentum)

    # performs backpropogation on a minibatch and sums outputs
    # batch - mini batch
    # class_outputs - contains a list of class outputs for classification problems
    # lr - learning rate of back propogation
    # momentum - momentum of back propogation
    def update_network(self, batch, class_outputs, lr, momentum):
        # sum of bias gradients
        d_b_sum = [np.zeros(b.shape) for b in self.biases]
        # sum of weight gradients
        d_w_sum = [np.zeros(w.shape) for w in self.weights]
        prev_delta = [np.zeros(w.shape) for w in self.weights]
        for i, arr in enumerate(batch):  
            
            # get gradients from test row       
            d_b, d_w = self.back_prop(arr[:-1], arr[-1], class_outputs)
            if(i == 0):
                prev_delta = d_w
            # add bias and weight gradients to their respective sums
            d_b_sum = [nb+dnb for nb, dnb in zip(d_b_sum, d_b)]
            d_w_sum = [nw+dnw for nw, dnw in zip(d_w_sum, d_w)]
            # print("dl/db sum:", d_b_sum)
            # print("dl/dw sum:", d_w_sum)
            # update d_w_sum to also add delta from previous iteration
            d_w_sum = [nw+momentum*(delta) for nw, delta in zip(d_w_sum, prev_delta)]
            # holds current iteration delta for next  iteration
            prev_delta = d_w
        # print("Weights Before: ", self.weights)
        # print("Biases Before: ", self.biases)
        # adjust weight and bias values using summed gradients
        self.weights = [w-(lr/len(batch))*nw for w, nw in zip(self.weights, d_w_sum)]
        self.biases = [b-(lr/len(batch))*nb for b, nb in zip(self.biases, d_b_sum)]
        # print("Weights After: ", self.weights)
        # print("Biases After: ", self.biases)
        
    
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

    # performs back propogation algorithm of a network
    # inp - is the input vector
    # expected - is the desired output
    # class_outputs - list of classes in the dataset
    # returns:
    # d_b - gradients of the bias vector
    # d_c - gradients of the weight vector
    def back_prop(self, inp, expected, class_outputs):
        d_w, d_b = [], []
        out, activations = self.feedforward(inp)
        
        err_dc_db, err_dc_dw = [], []
        if self.problemType == "classification":
            # one hot encoding array based on expected output
            expected = self.one_hot_encoding(expected, class_outputs)
            # softmax partial derivative in regards to bias
            err_dc_db = self.cross_entropy_drvt(activations[-1], expected)
            d_b.insert(0,err_dc_db) 
            # softmax partial derivative in regards to weight
            err_dc_dw = np.dot(err_dc_db, activations[-2].transpose())
            d_w.insert(0,err_dc_dw)           
        else: 
            # linear partial derivative in regards to bias
            err_dc_db = activations[-1] - expected
            d_b.insert(0,err_dc_db) 
            # linear partial derivative in regards to weight
            err_dc_dw = err_dc_db * activations[-2].T
            d_w.insert(0,err_dc_dw)
        # print("Output: ", out)
        # print("Target: ", expected)
        # print("dl/db:", err_dc_db)
        # print("Layer Inputs: ", activations[-2].transpose())
        # print("dl/dw:", err_dc_dw)
        for l in range(2,len(self.net_props)):
            err_dc_db = np.dot(self.weights[-l+1].transpose(), err_dc_db) * self.sigmoid_drvt(activations[-l]) 
            d_b.insert(0,err_dc_db)
            err_dc_dw = np.dot(err_dc_db, activations[-l-1].transpose())
            d_w.insert(0,err_dc_dw)
        return d_b, d_w


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
