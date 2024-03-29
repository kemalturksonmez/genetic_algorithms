# Authors:
# Kemal Turksonmez
# Arash Ajam
from network import Network
from backpropogation import BackPropogation
from diffevolution import DE
from genetic import GA
from process_data import PD
from genetic import GA
import numpy as np
from particleSwarm import PSO
import matplotlib.pyplot as plt
''' This class contains methods to run and train models on each data set '''
class Models:
    def __init__(self):
        self.pd = PD()

    # performs cross validation on a given dataset and method
    # programType - method of tuning
    # fileName - name of dataset
    # data - data
    # net - network object
    # batch_size - size of a mini batch
    # num_runs - number of generations/runs
    # backPropParam - back propogation parameters
    # deParam - differential evolution parameters
    # psoParam - particle swarm optimization parameters
    # gaParam - genetic algorithm parameters
    # verbose - boolean used to print out results
    def graphData(self, programType, fileName, data, net, classOutputs, batch_size, num_runs, backPropParam, deParam, psoParam, gaParam, bestIndex):
        # dataset split
        if net.problemType == "classification":
            trainData, testData = self.pd.stratifiedSplit(data, bestIndex)
        else:
            trainData, testData = self.pd.regressiveSplit(data, bestIndex) 
        
        if programType == "BP":
            bp = BackPropogation(net)
        elif programType == "DE":
            de = DE(net)
        elif programType == "PSO":
            pso = PSO(net)
        elif programType == "GA":
            ga = GA(net)
        # graph network
        size = 400
        numPoints = int(num_runs/size)
        lossArr = []
        for i in range(numPoints):
            loss = 0
            if programType == "BP":
                bp.stochastic_GD(trainData, classOutputs, batch_size, size, backPropParam[0], backPropParam[1])
            elif programType == "DE":
                de.train(trainData, deParam[0], deParam[1], classOutputs, size, batch_size)
            elif programType == "PSO":
                pso.train(trainData, psoParam[0], psoParam[1], psoParam[2], psoParam[3], classOutputs, num_runs, batch_size)
            elif programType == "GA":
                ga.train(trainData, gaParam[0], gaParam[1], classOutputs, num_runs, batch_size)

            loss, acc = net.get_accuracy(testData, classOutputs)
            lossArr.append(loss/len(testData))

        error_name = "Cross Entropy Loss"
        if net.problemType == "regression":
            error_name = " Mean Squared Error"
        plt.plot(lossArr)
        plt.xlabel("Training Iteration [Multiplied by " + str(size) + "]")
        plt.ylabel(error_name)
        plt.title('Error vs Training Iteration on ' + str(fileName) + " With " + str(net.n_hidden) + " Layers", fontsize=12)
        plt.savefig('results/' + fileName + '_' + programType + '_' + str(net.n_hidden) +'_layers_loss.png', dpi=600, bbox_inches='tight')
        plt.clf()

    # performs cross validation on a given dataset and method
    # programType - method of tuning
    # fileName - name of dataset
    # data - data
    # net - network object
    # batch_size - size of a mini batch
    # num_runs - number of generations/runs
    # backPropParam - back propogation parameters
    # deParam - differential evolution parameters
    # psoParam - particle swarm optimization parameters
    # gaParam - genetic algorithm parameters
    # verbose - boolean used to print out results
    def cross_validation(self, programType, fileName, data, net, batch_size, num_runs, backPropParam, deParam, psoParam, gaParam, verbose=True):
        # standardize data
        data = self.pd.standardize_data(data)
        # number of folds in k-folds cross validation
        cv_num = 10
        # class outputs contains all the potential classes in a classification data set
        classOutputs = []
        # get all the potential classes
        if net.problemType == "classification":
            data = self.pd.split_by_class(data)
            for key in data:
                classOutputs.append(key)
        # convert each element to it's own array
        data = self.pd.convert_to_single_numpy(data, net.problemType)
        beforeAccSum, beforeLossSum, beforeBestIndex = 0, 0, 0
        afterAccSum, afterLossSum, afterBestIndex = 0, 0, 0
        afterBestLoss = float('inf')
        beforeBestLoss = float('inf')
        beforeBestAcc = 0
        afterBestAcc = 0
        error_name = "Cross Entropy: "
        if net.problemType == "regression":
            error_name = "Mean Squared Error: "
        # get array of outputs
        for i in range(cv_num):
            if verbose:
                print("Running iteration: ", i)
            # dataset split
            if net.problemType == "classification":
                trainData, testData = self.pd.stratifiedSplit(data, i)
            else:
                trainData, testData = self.pd.regressiveSplit(data, i) 
           
            if verbose:
                print("Outputs before training:")
            loss, acc = net.get_accuracy(testData, classOutputs)
            if net.problemType == "classification":
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
            if programType == "BP":     
                backprop = BackPropogation(net)
                backprop.stochastic_GD(trainData, classOutputs, batch_size, num_runs, backPropParam[0], backPropParam[1])
            # train DE
            elif programType == "DE":
                de = DE(net)
                de.train(trainData, deParam[0], deParam[1], classOutputs, num_runs, batch_size)
            # train PSO
            elif programType == "PSO":
                pso = PSO(net)
                pso.train(trainData, psoParam[0], psoParam[1], psoParam[2], psoParam[3], classOutputs, num_runs, batch_size)
            # train GA
            elif programType == "GA":
                ga = GA(net)
                ga.train(trainData, gaParam[0], gaParam[1], classOutputs, num_runs, batch_size)

            if verbose:
                print("Outputs after training:")
            loss, acc = net.get_accuracy(testData, classOutputs)

           
            if net.problemType == "classification":
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
            # create network object
            net = Network(net.n_hidden, net.n_outputs, net.n_inputs, net.layer_nodes, net.problemType, verbose)
        
        # self.graphData(programType, fileName, data, net, classOutputs, batch_size, num_runs, backPropParam, deParam, psoParam, gaParam, bestIndex)
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

    ################ breast cancer
    def cancer(self, programType):
        data = 'data/breast-cancer-wisconsin.data'
        with open(data) as inp:
            text = [l.replace("?", "-1") for l in inp]
        data = np.loadtxt(text, delimiter=",")
        # remove id
        data = self.pd.remove_first_column(data)
        # remove missing rows
        data = self.pd.remove_missing(data)
        # Run cross validation
        problemType = "classification"
        fileName = "cancer"
        n_inputs = 9
        n_outputs = 2
        batch_size = 21
        # num_runs = 8000
        num_runs = 1000
        verbose = True
        # BP
        lr = 0.008
        momentum = 0.0001   
        # DE   
        beta = 0.07
        cross_prob= 0.90
        # PSO
        omega = 0.001
        cog_1 = 0.01
        cog_2 = 0.1
        alpha = 0.3
        # GA
        mutate_p = .0005
        selection_rate = .2

        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 1
        layer_nodes = 6
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 2
        layer_nodes = 4
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)


    ################ glass
    def glass(self, programType):
        data = 'data/glass.data'
        data = np.loadtxt(data, delimiter=",")
        # remove id
        data = self.pd.remove_first_column(data)
        # Run cross validation
        problemType = "classification"
        fileName = "glass"
        n_inputs = 9
        n_outputs = 6
        batch_size = 20
        # num_runs = 10000
        num_runs = 500
        verbose = True
        # BP
        lr = 0.008
        momentum = 0.002      
        # DE
        beta = 0.12
        cross_prob= 0.90
        # PSO
        omega = 0.15
        cog_1 = 0.01
        cog_2 = 0.1
        alpha = 0.3
        # GA
        mutate_p = .0001
        selection_rate = .2
        
        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 1
        layer_nodes = 7
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 2
        layer_nodes = 4
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        # self.cross_validation(data, "classification", "glass", 0, 6, 9, 0, 8, 20000, 0.008, 0.002)
        # self.cross_validation(data, "classification", "glass", 1, 6, 9, 7, 8, 20000, 0.008, 0.002)
        # self.cross_validation(data, "classification", "glass", 2, 6, 9, 4, 8, 20000, 0.008, 0.002)


    ################ soybean
    def soybean(self, programType):
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
        batch_size = 15
        # num_runs = 4000
        num_runs = 1000
        verbose = True
        # BP
        lr = 0.001
        momentum= 0.001   
        # DE   
        beta = 0.18
        cross_prob= 0.9
        # PSO
        omega = 0.001
        cog_1 = 0.01
        cog_2 = 0.1
        alpha = 0.3
        # GA
        mutate_p = .0001
        selection_rate = .2

        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)
        
        n_hidden = 1
        layer_nodes = 20
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)
        
        n_hidden = 2
        layer_nodes = 15
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)



    ################ Abalone
    def abalone(self, programType):
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
        problemType = "regression"
        fileName = "abalone"
        n_inputs = 8
        n_outputs = 1
        batch_size = 55
        num_runs = 5000
        verbose = True
        # BP
        lr = 0.001
        momentum= 0.01      
        # DE
        beta = 0.16
        cross_prob= 0.9
        # PSO
        omega = 0.001
        cog_1 = 0.01
        cog_2 = 0.1
        alpha = 0.3
        # GA
        mutate_p = .00001
        selection_rate = .2

        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 1
        layer_nodes = 5
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p],verbose)

        n_hidden = 2
        layer_nodes = 3
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p],verbose)

        # self.cross_validation(data, "regression", "abalone", 0, 1, 8, 0, 55, 10000, 0.00005, 0.01)
        # self.cross_validation(data, "regression", "abalone", 1, 1, 8, 5, 55, 10000, 0.00005, 0.01)
        # self.cross_validation(data, "regression", "abalone", 2, 1, 8, 3, 55, 10000, 0.00005, 0.01)


    ################ Computer Hardware
    def hardware(self, programType):
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
        problemType = "regression"
        fileName = "hardware"
        n_inputs = 6
        n_outputs = 1
        batch_size = 35
        num_runs = 20000
        verbose = True
        # BP
        lr = 0.00005
        momentum= 0.01      
        # DE
        beta = 0.16
        cross_prob= 0.9
       # PSO
        omega = .6
        cog_1 = 0.2
        cog_2 = 0.6
        alpha = 0.99
        # GA
        mutate_p = .00001
        selection_rate = .2
        
        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p],verbose)

        n_hidden = 1
        layer_nodes = 8
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p],verbose)

        n_hidden = 2
        layer_nodes = 4
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p],verbose)
        
        # self.cross_validation(data, "regression", "machine", 0, 1, 6, 0, 35, 10000, 0.00005, 0.01)
        # self.cross_validation(data, "regression", "machine", 1, 1, 6, 8, 35, 10000, 0.00005, 0.01)
        # self.cross_validation(data, "regression", "machine", 2, 1, 6, 4, 35, 10000, 0.00005, 0.01)

    
    ############### Forest Fires
    def fires(self, programType):
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
        # Run cross validation
        problemType = "regression"
        fileName = "forestfires"
        n_inputs = 14
        n_outputs = 1
        batch_size = 10
        num_runs = 20000
        verbose = True
        # BP
        lr = 0.00005
        momentum= 0.006      
        # DE
        beta = 0.16
        cross_prob= 0.9
        # PSO
        omega = 0.01
        cog_1 = 0.01
        cog_2 = 0.8
        alpha = 0.93
        # GA
        mutate_p = .00001
        selection_rate = .2
        
        n_hidden = 0
        layer_nodes = 0
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 1
        layer_nodes = 6
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        n_hidden = 2
        layer_nodes = 4
        net = Network(n_hidden, n_outputs, n_inputs, layer_nodes, problemType, verbose)
        self.cross_validation(programType, fileName, data, net, batch_size, num_runs, [lr, momentum], [beta, cross_prob], [omega, cog_1, cog_2, alpha], [selection_rate, mutate_p], verbose)

        # # self.cross_validation(data, "regression", "forestfires", 1, 1, 14, 6, 7, 8000, 0.00005, 0.02)
        # self.cross_validation(data, "regression", "forestfires", 0, 1, 14, 0, 3, 12000, 0.00005, 0.006)
        # self.cross_validation(data, "regression", "forestfires", 1, 1, 14, 6, 3, 12000, 0.00005, 0.006)
        # self.cross_validation(data, "regression", "forestfires", 2, 1, 14, 4, 3, 12000, 0.00002, 0.006)