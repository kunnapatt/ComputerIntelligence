import numpy as np
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

#### Attributes | 3, 6, 8, 10, 11, 12, 13, 14 to input
#### Attributes | 5 to output

def main():
    input_x, true_x = getData()

    input_x = transposeList(input_x)

    x_train, y_train, x_test, y_test= createFold(10, input_x, true_x)


    for i in range(len(x_train)): #### Fold
        print("Fold {}" .format(i+1))
        
        neural = NeuralNetwork(x_train[i], y_train[i])
        neural.addLayer(70)
        # neural.addLayer(10)
        # neural.addLayer(5)
        # neural.addLayer(10)
        neural.addLayer(1)
        
        neural.fit(50, 100)

        neural.plotFitness(i+1)
        neural.evaluate(x_test[i], y_test[i])

        print("---------------")

def getData(): ### return list
    path = 'AirQualityUCI.xlsx'
    listdata = []
    
    df = pd.read_excel(path)
    x = []
    cou = 0

    for atr in df.columns:
        if cou == 3 or cou == 6 or cou == 8 or cou >= 10 :
            x.append(normalizeList(df[atr]))
        elif cou == 5 :
            y = normalizeList(df[atr]) 
        cou += 1
    
    return x, y

def transposeList(datalist):
    return list(map(list, zip(*datalist)))

def normalizeList(x):
    # x = transposeList(x)
    nor_x = []

    minx = min(x)
    maxx = max(x)

    for i in range(len(x)):
        nor = (x[i] - minx)/(maxx - minx)
        nor_x.append(nor)
    return nor_x

class NeuralNetwork:
    def __init__(self, inpu, out):
        # np.random.seed(1)
        self.Fullinput = inpu
        self.FullTrueOutput = out
        self.input = np.asarray(copy.deepcopy(inpu))
        self.weight = []
        self.bias = []
        self.predict = 0
        self.lr = 0
        self.E = 0
        self.countLayer = 0
        self.Node = []
        self.output = []
        self.TrueOutput= None
        self.cou = 0
        self.deltaweight = []
        self.err = []
        self.deltabias = []
        self.gradient = [] # back to front
        self.momentum = 0
        self.Loss = []
        self.minimumLoss = 10000000
        self.rememberweightT_1 = []
        self.countsample = 0
        self.rememberweightT_2 = []
        self.npopulation = 0
        self.group_pop = []
        self.pbest = []
        self.gbest = [[100000, 0]]
        self.velocity = []

    def addLayer(self, node):
        self.countLayer+=1
        self.Node.append(node)

    def sigmoid(self, v):
        return 1/(1+np.exp(-v))

    def diffsigmoid(self, y):
        return y*(1-y)

    def createweight(self):
        for i in range(len(self.Node)):
            if ( i == 0 ):
                self.weight.append(2*np.random.rand(len(self.input), self.Node[i]) - 1)
                self.bias.append(2*np.random.rand(self.Node[i]) - 1)
                # self.deltaweight.append(np.ones((len(self.input), self.Node[i])))
                # self.deltabias.append(np.ones(self.Node[i]))
                # self.rememberweightT_1.append(np.zeros((len(self.input), self.Node[i])))
                # self.rememberweightT_2.append(np.zeros((len(self.input), self.Node[i])))
            else:
                self.weight.append(2*np.random.rand(self.Node[i-1], self.Node[i]) - 1)
                self.bias.append(2*np.random.rand(self.Node[i]) - 1)
                # self.deltaweight.append(np.ones((self.Node[i-1], self.Node[i])))
                # self.deltabias.append(np.ones(self.Node[i]))
                # self.rememberweightT_1.append(np.zeros((len(self.input), self.Node[i])))
                # self.rememberweightT_2.append(np.zeros((len(self.input), self.Node[i])))

    def FeedForward(self, chromosome): # each sample, each generation
        self.output = []
        self.output.append(self.input.T)
        out = np.array(self.output[0])
        for i in range(len(self.Node)): # feed in each layer
            v = np.dot(copy.deepcopy(chromosome[i].T), copy.deepcopy(self.output[i]))
            out = self.sigmoid(v)
            self.output.append(out)
        
        return out

    def fit(self, npopulatation, iteration):
        self.npopulation = npopulatation
        self.fitnessplot = []
        for k in range(iteration) :
            fitne = []
            for i in range(npopulatation): # individual
                if k == 0 :
                    temp = self.population()
                    temp = np.asarray(copy.deepcopy(temp))
                    out = self.FeedForward(temp)

                else :
                    out = self.FeedForward(copy.deepcopy(self.group_pop[i][1]))

                self.Loss = []

                for j in range(len(self.Fullinput)): # sample
                    err = out[0][j] - copy.deepcopy(self.FullTrueOutput[j])
                    self.Loss.append(err)
                mae = np.asarray(copy.deepcopy(self.Loss))
                
                fitness = self.fitness(mae)
                fitne.append(fitness)

                #### compare find pbest and gbest
                if k == 0 :
                    self.group_pop.append([fitness, temp])
                    self.velocity.append(copy.deepcopy(self.population()))
                    self.pbest.append(copy.deepcopy(self.group_pop[i]))
                # else :
                if fitness < self.pbest[i][0] : ### best of self
                    self.pbest[i][0] = copy.deepcopy(fitness)
                    self.pbest[i][1] = copy.deepcopy(self.group_pop[i][1])

                if  fitness < self.gbest[0][0] : ### best of global
                    self.gbest[0][0] = copy.deepcopy(fitness)
                    self.gbest[0][1] = copy.deepcopy(self.group_pop[i][1])

                # update velocity
                if k + i == 0 :
                    rho1, rho2 = self.rho()

                self.velocity[i] = copy.deepcopy(self.velocity[i]) + rho1*(copy.deepcopy(self.pbest[i][1]) - copy.deepcopy(self.group_pop[i][1])) + rho2*(copy.deepcopy(self.gbest[0][1]) - copy.deepcopy(self.group_pop[i][1]))

                # update position
                self.group_pop[i][1] = copy.deepcopy(self.group_pop[i][1]) + copy.deepcopy(self.velocity[i])
            
            self.fitnessplot.append(np.mean(np.asarray(copy.deepcopy(fitne))))

    def population(self): # return list
        weight = []
        for  i in range(len(self.Node)):
            if (i == 0):
                weight.append(np.random.uniform(-1, 1, (len(self.Fullinput[0]), self.Node[i])))
            else : 
                weight.append(np.random.uniform(-1, 1, (self.Node[i-1], self.Node[i])))
            
        return weight

    def fitness(self, mae): #### Cal fitness from error | input Integer | how to fitness maximum
        return np.mean(np.abs(mae))

    def evaluate(self, x_test, y_test):
        x_test = np.asarray(copy.deepcopy(x_test))
        y_test = np.asarray(copy.deepcopy(y_test))

        self.input = x_test
        fitne = []
        for i in range(len(self.group_pop)):

            predict = self.FeedForward(self.group_pop[i][1])

            Loss = []
            for j in range(len(y_test)):
                err = predict[0][j] - y_test[j]
                Loss.append(err)

            mse = np.asarray(copy.deepcopy(Loss))
            mse = np.power(mse, 2)/ 2
            mse = np.mean(mse) # scalar
            fitness = self.fitness(mse)
            fitne.append(fitness)

        fitne = np.asarray(copy.deepcopy(fitne))
        fitne = np.mean(fitne)
        print("Fitness = {}" .format(fitne))                

    def plotFitness(self, fold):
        fitness = self.fitnessplot
        fig, ax = plt.subplots()
        ax.plot(range(1, len(fitness)+1), fitness)
        ax.set(xlabel='Iteration', ylabel='Fitness', title='Fold {}' .format(fold))
        fig.savefig("Fold {}.png" .format(fold))

    def rho(self):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        c1 = np.random.uniform(0, 2)
        c2 = np.random.uniform(0, 2)

        rho1 = r1*c1
        rho2 =  r2*c2

        return rho1, rho2

def createFold(fold, x_data, y_data):
    crossvalidation = int(len(x_data)*fold/100)
    # print(len(x_data))
    x_train_testingset = []
    y_true_testingset = []

    x_train_trainingset = []
    y_true_trainingset = []

    for i in range(fold):
        if i == (fold-1) :
            x_train_testingset.append(x_data[0+i*crossvalidation:len(x_data)])
            y_true_testingset.append(y_data[0+i*crossvalidation:len(y_data)])

            x_train_trainingset.append(x_data[0:0+i*crossvalidation])
            y_true_trainingset.append(y_data[0:0+i*crossvalidation])
        else:
            x_train_testingset.append(x_data[0+i*crossvalidation:crossvalidation+i*crossvalidation])
            y_true_testingset.append(y_data[0+i*crossvalidation:crossvalidation+i*crossvalidation])

            x_train_trainingset1 = x_data[0:i*crossvalidation]
            x_train_trainingset2 = x_data[crossvalidation*(i+1):len(x_data)]

            x_train_trainingset.append(x_train_trainingset1 + x_train_trainingset2)

            y_true_trainingset1 = y_data[0:i*crossvalidation]
            y_true_trainingset2 = y_data[crossvalidation*(i+1):len(x_data)]

            y_true_trainingset.append(y_true_trainingset1 + y_true_trainingset2)

    return x_train_trainingset, y_true_trainingset, x_train_testingset, y_true_testingset

if __name__ == "__main__":
    main()