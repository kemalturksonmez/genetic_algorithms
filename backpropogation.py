import numpy as np
from copy import deepcopy
from numpy.random import shuffle
import random
class BackPropogation:
    # Creates a network object
    # network - network
    def __init__(self, network):
        self.net = network

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
        # sum of weight gradients
        d_w_sum = [np.zeros(w.shape) for w in self.net.weights]
        prev_delta = [np.zeros(w.shape) for w in self.net.weights]
        for i, arr in enumerate(batch):  
            
            # get gradients from test row       
            d_w = self.back_prop(arr[:-1], arr[-1], class_outputs)
            if(i == 0):
                prev_delta = d_w
            # add bias and weight gradients to their respective sums
            d_w_sum = [nw+dnw for nw, dnw in zip(d_w_sum, d_w)]
            # update d_w_sum to also add delta from previous iteration
            d_w_sum = [nw+momentum*(delta) for nw, delta in zip(d_w_sum, prev_delta)]
            # holds current iteration delta for next  iteration
            prev_delta = d_w
        # adjust weight and bias values using summed gradients
        self.net.weights = [w-(lr/len(batch))*nw for w, nw in zip(self.net.weights, d_w_sum)]

    # performs back propogation algorithm of a network
    # inp - is the input vector
    # expected - is the desired output
    # class_outputs - list of classes in the dataset
    # returns:
    # d_b - gradients of the bias vector
    # d_c - gradients of the weight vector
    def back_prop(self, inp, expected, class_outputs):
        d_w = []
        out, activations = self.net.feedforward(inp)
        
        err_dc_dw = []
        if self.net.problemType == "classification":
            # one hot encoding array based on expected output
            expected = self.net.one_hot_encoding(expected, class_outputs)
            # softmax partial derivative in regards to bias
            err_dc_db = self.net.cross_entropy_drvt(activations[-1], expected)
            # softmax partial derivative in regards to weight
            err_dc_dw = np.dot(err_dc_db, activations[-2].transpose())
            d_w.insert(0,err_dc_dw)           
        else: 
            # linear partial derivative in regards to bias
            err_dc_db = activations[-1] - expected
            # linear partial derivative in regards to weight
            err_dc_dw = err_dc_db * activations[-2].T
            d_w.insert(0,err_dc_dw)
        for l in range(2,len(self.net.net_props)):
            err_dc_db = np.dot(self.net.weights[-l+1].transpose(), err_dc_db) * self.net.sigmoid_drvt(activations[-l]) 
            err_dc_dw = np.dot(err_dc_db, activations[-l-1].transpose())
            d_w.insert(0,err_dc_dw)
        return d_w

