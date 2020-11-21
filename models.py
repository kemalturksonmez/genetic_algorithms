# Authors:
# Kemal Turksonmez
# Arash Ajam
from network import Network
from backpropogation import BackPropogation
from diffevolution import DE
from process_data import PD
import numpy as np
# This class contains methods to run and train models on each data set
class Models:
    def __init__(self):
        self.pd = PD()

    def graphData(self, n_hidden, n_outputs, n_inputs, layer_nodes, problemType, classOutputs, batch_size, num_runs, lr, momentum, bestIndex):
        # dataset split
        if problemType == "classification":
            trainData, testData = self.pd.stratifiedSplit(data, bestIndex)
        else:
            trainData, testData = self.pd.regressiveSplit(data, bestIndex) 
        # create new network object
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType)
        backprop = BackPropogation(net)
        backprop.stochastic_GD(trainData, classOutputs, batch_size, num_runs, lr, momentum)

        # graph network
        net.graph(fileName, trainData, testData, classOutputs, batch_size, num_runs, lr, momentum)

    def cross_validation(self, data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, beta=0, cross_prob=0, verbose=True):
        # standardize data
        data = self.pd.standardize_data(data)
        # number of folds in k-folds cross validation
        cv_num = 10
        # class outputs contains all the potential classes in a classification data set
        classOutputs = []
        # get all the potential classes
        if problemType == "classification":
            data = self.pd.split_by_class(data)
            for key in data:
                classOutputs.append(key)
        # convert each element to it's own array
        data = self.pd.convert_to_single_numpy(data, problemType)
        beforeAccSum, beforeLossSum, beforeBestIndex = 0, 0, 0
        afterAccSum, afterLossSum, afterBestIndex = 0, 0, 0
        afterBestLoss = float('inf')
        beforeBestLoss = float('inf')
        beforeBestAcc = 0
        afterBestAcc = 0
        error_name = "Cross Entropy: "
        if problemType == "regression":
            error_name = "Mean Squared Error: "
        # get array of outputs
        for i in range(cv_num):
            if verbose:
                print("Running iteration: ", i)
            # dataset split
            if problemType == "classification":
                trainData, testData = self.pd.stratifiedSplit(data, i)
            else:
                trainData, testData = self.pd.regressiveSplit(data, i) 
           
            # network object
            net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
            if verbose:
                print("Outputs before training:")
            loss, acc = net.get_accuracy(testData, classOutputs)
            if problemType == "classification":
                if verbose:
                    print("Accuracy: ", acc)
                beforeAccSum += acc
                if acc > beforeBestAcc:
                    beforeBestAcc = acc
            beforeLossSum += loss
            if loss < beforeBestLoss:
                beforeBestLoss = loss    
            if verbose:    
                print(error_name, loss)
            # train back propogation
            # backprop = BackPropogation(net)
            # backprop.stochastic_GD(trainData, classOutputs, batch_size, num_runs, lr, momentum)

            # train DE
            de = DE(net)
            de.train(trainData, beta, cross_prob, classOutputs, num_runs, batch_size, loss)
            if verbose:
                print("Outputs after training:")
            loss, acc = net.get_accuracy(testData, classOutputs)

           
            if problemType == "classification":
                if verbose:
                    print("Accuracy: ", acc)
                afterAccSum += acc
                if acc > afterBestAcc:
                    afterBestAcc = acc
            afterLossSum += loss
            if loss < afterBestLoss:
                afterBestLoss = loss
                bestIndex = i
            if verbose:
                print(error_name, loss)
                print()
        
        # self.graphData(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, classOutputs, batch_size, num_runs, lr, momentum, bestIndex)
        if verbose:
            print("Before Accuracy average:", beforeAccSum/cv_num)
            print("Before Loss average:", beforeLossSum/cv_num)
            print("Before Best Loss: ", beforeBestLoss)
            print("Before Best accuracy:", beforeBestAcc)
            print("After Accuracy average:", afterAccSum/cv_num)
            print("After Loss average:", afterLossSum/cv_num)
            print("After Best Loss: ", afterBestLoss)
            print("After Best accuracy:", afterBestAcc)
            print()
        return afterLossSum/cv_num

    def tuneDE(self, data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, beta_range, cross_range):
        best_beta = 1
        best_cross_prob = 0.5
        best_loss = self.cross_validation(data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, best_beta, best_cross_prob, False)
        for i in range(3):
            for j in range(3):
                beta = ((beta_range[1]- beta_range[0])/3) * i +  beta_range[0]
                cross_prob = ((cross_range[1]-cross_range[0])/3) * j + cross_range[0]
                loss = self.cross_validation(data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, beta, cross_prob, False)
                if loss < best_loss:
                    best_beta = beta
                    best_cross_prob = cross_prob
        print("Beta Range", beta_range)
        print("Best Beta: ", best_beta)
        print("Cross Range", cross_range)
        print("Best Cross Prob: ", best_cross_prob)
        print("Best Loss Avg Avg: ", best_loss)
        beta_range = (beta_range[1] - beta_range[0])/3
        cross_prob = (cross_range[1] - cross_range[0])/3
        self.tuneDE(data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, [best_beta - beta_range,best_beta + beta_range], [best_cross_prob - cross_prob,best_cross_prob + cross_prob])

    ################ breast cancer
    def cancer(self):
        data = 'data/breast-cancer-wisconsin.data'
        with open(data) as inp:
            text = [l.replace("?", "-1") for l in inp]
        data = np.loadtxt(text, delimiter=",")
        # remove id
        data = self.pd.remove_first_column(data)
        # remove missing rows
        data = self.pd.remove_missing(data)
        # Run cross validation
        # self.cross_validation(data, "classification", "cancer", 2, 2, 9, 4, 7, 40000, 0.002, 0.001)

        self.cross_validation(data, "classification", "cancer", 0, 2, 9, 0, 7, 40000, 0.008, 0.0001)
        self.cross_validation(data, "classification", "cancer", 1, 2, 9, 6, 7, 40000, 0.008, 0.0001)
        self.cross_validation(data, "classification", "cancer", 2, 2, 9, 4, 7, 40000, 0.008, 0.0001)

    ################ glass
    def glass(self):
        data = 'data/glass.data'
        data = np.loadtxt(data, delimiter=",")
        # remove id
        data = self.pd.remove_first_column(data)
        # Run cross validation
        # self.cross_validation(data, "classification", "glass", 2, 6, 9, 7, 8, 20000, 0.005, 0.002)

        self.cross_validation(data, "classification", "glass", 0, 6, 9, 0, 8, 20000, 0.008, 0.002)
        self.cross_validation(data, "classification", "glass", 1, 6, 9, 7, 8, 20000, 0.008, 0.002)
        self.cross_validation(data, "classification", "glass", 2, 6, 9, 4, 8, 20000, 0.008, 0.002)


    ################ soybean
    def soybean(self):
        data = 'data/soybean-small.data'
        text = []
        with open(data) as inp:
                for l in inp:
                    l = l.replace("D1", "1")
                    l = l.replace("D2", "2")
                    l = l.replace("D3", "3")
                    l = l.replace("D4", "4")
                    text.append(l)
        data = np.loadtxt(text, delimiter=",")
        problemType = "classification"
        fileName = "soybean"
        n_inputs = 35
        n_outputs = 4
        num_runs = 10000
        # self.cross_validation(data, "classification", "soybean", 2, 4, 35, 15, 4, 25, 1)

        # self.cross_validation(data, "classification", "soybean", 0, 4, 35, 0, 3, 10000, 0.001, 0.001)
        # self.cross_validation(data, "classification", "soybean", 1, 4, 35, 20, 3, 10000, 0.001,0.001)
        # self.cross_validation(data, "classification", "soybean", 2, 4, 35, 15, 3, 10000, 0.001, 0.001)
        
        n_hidden = 2
        layer_nodes = 8
        batch_size = 15
        num_runs = 400
        lr = 0.001
        momentum= 0.001
        beta = 1
        cross_prob= 0.5
        self.tuneDE(data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, [0,2], [0.8,1])
        # self.cross_validation(data, problemType, fileName, n_hidden, n_outputs, n_inputs, layer_nodes, batch_size, num_runs, lr, momentum, beta, cross_prob)


    ################ Abalone
    def abalone(self):
        data = 'data/abalone.data'
        IntToClass = {"0" : "M", "1" : "F", "2" : "I"}
        text = []
        with open(data) as inp:
                for l in inp:
                    l = l.replace("M", "0")
                    l = l.replace("F", "1")
                    l = l.replace("I", "2")
                    l = l.replace("n", "0")
                    # l = l.replace("?", "-1")
                    text.append(l)
        data = np.loadtxt(text, delimiter=",")
        # place regression values at the end
        data = self.pd.shift_first_column(data)
        # # remove missing rows
        # data = self.pd.remove_missing(data)
        # Run cross validation
        # self.cross_validation(data, "regression", "abalone", 1, 1, 8, 5, 55, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "abalone", 0, 1, 8, 5, 55, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "abalone", 1, 1, 8, 5, 55, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "abalone", 2, 1, 8, 3, 55, 10000, 0.00005, 0.01)


    ################ Computer Hardware
    def hardware(self):
        data = 'data/machine.data'
        data = np.genfromtxt(data,dtype='str',delimiter=",")
        # remove non-predictive column
        data = self.pd.remove_first_column(data)
        # remove non-predictive column
        data = self.pd.remove_first_column(data)
        # seperate last column which includes predictions from previous research project
        data, deleted = self.pd.remove_last_column(data)
        data = np.asarray(data, dtype='float')
        deleted = np.asarray(deleted, dtype='float')
        # Run cross validation
        # self.cross_validation(data, "regression", "machine", 1, 1, 6, 8, 35, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "machine", 0, 1, 6, 0, 35, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "machine", 1, 1, 6, 8, 35, 10000, 0.00005, 0.01)
        self.cross_validation(data, "regression", "machine", 2, 1, 6, 4, 35, 10000, 0.00005, 0.01)

    
    ############### Forest Fires
    def fires(self):
        data = 'data/forestfires.data'
        text = []
        # adjust to make cyclic
        with open(data) as inp:
        		for l in inp:
        			l = l.replace("jan", "0")
        			l = l.replace("feb", "1")
        			l = l.replace("mar", "2")
        			l = l.replace("apr", "3")
        			l = l.replace("may", "4")
        			l = l.replace("jun", "5")
        			l = l.replace("jul", "6")
        			l = l.replace("aug", "7")
        			l = l.replace("sep", "8")
        			l = l.replace("oct", "9")
        			l = l.replace("nov", "10")
        			l = l.replace("dec", "11")
        			l = l.replace("sun", "0")
        			l = l.replace("mon", "1")
        			l = l.replace("tue", "2")
        			l = l.replace("wed", "3")
        			l = l.replace("thu", "4")	
        			l = l.replace("fri", "5")
        			l = l.replace("sat", "6")
        			text.append(l)
        data = np.loadtxt(text, delimiter=",")
        # Run cross validation
        # seperate each cyclic feature into two seperate features
        tempData = []
        for row in data:
            tempRow = []
            for i in range(len(row)):
                if i == 2:
                    sin, cos = self.pd.convert_to_cyclic(row[i], 12)
                    tempRow.append(sin)
                    tempRow.append(cos)
                elif i == 3:
                    sin, cos = self.pd.convert_to_cyclic(row[i], 7)
                    tempRow.append(sin)
                    tempRow.append(cos)
                else:
                    tempRow.append(row[i])
            tempData.append(tempRow)
        data = tempData
        # self.cross_validation(data, "regression", "forestfires", 1, 1, 14, 6, 7, 8000, 0.00005, 0.02)
        self.cross_validation(data, "regression", "forestfires", 0, 1, 14, 0, 3, 12000, 0.00005, 0.006)
        self.cross_validation(data, "regression", "forestfires", 1, 1, 14, 6, 3, 12000, 0.00005, 0.006)
        self.cross_validation(data, "regression", "forestfires", 2, 1, 14, 4, 3, 12000, 0.00002, 0.006)