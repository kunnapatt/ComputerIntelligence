import numpy as np
import copy
import csv
#### data | 1). ID Number, 2). Diagnosis (M = malignant, B = benign) -> Class 3). 3 -> 32 is features ####

def main():
    data = getData()

    data = normalizeList(data)

    input_x = data[1:] # 2 dim for each features and sample
    true_x = data[0] # 1 dim for each sample

    neural = NeuralNetwork(input_x, true_x)
    neural.addLayer(5)
    neural.addLayer(10)
    neural.addLayer(1)
    neural.fit(2, 5)
    # a = neural.gene()

    # print(len(a[0]))
    # print(len(a[1]))
    # print(len(a[2]))

    # print(neural.Node)

    # print(a[0])


def getData(): ### return list
    path = 'wdbc.data.txt'
    listdata = []
    with open(path, 'r') as fi:
        row = [line for line in fi.read().splitlines()]

    for r in row :
        listdata.append(r.split(','))
    
    new = []
    for i in range(len(listdata)) :
        new.append([])
        for j in range(len(listdata[i])) :
            if listdata[i][j] == 'M' :
                new[i].append(0)
            elif listdata[i][j] == 'B' :
                new[i].append(1)
            else :
                new[i].append(float(listdata[i][j]))

    return new

def transposeList(datalist):
    return list(map(list, zip(*datalist)))

def normalizeList(x):
    x = transposeList(x)
    nor_x = []
    nor_x.append(x[1])
    for i in range(2, len(x)):
        nor_x.append([])
        minx = min(x[i])
        maxx = max(x[i])
        for j in range(len(x[i])):
            nor = (x[i][j] - minx)/(maxx - minx)
            nor_x[i-1].append(nor)

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
        self.individual = 0
        self.chromosome = []
        
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

    def FeedForward(self, chromosome):
        # each sample, each generation
        self.output = []
        self.output.append(self.input.T)
        out = np.array(self.output[0])
        for i in range(len(self.Node)): # feed in each layer
            # v = np.dot(self.output[i], chromosome[i]) + self.bias[i]
            v = np.dot(copy.deepcopy(self.output[i]), copy.deepcopy(chromosome[i]))
            out = self.sigmoid(v)
            self.output.append(out)
        
        return out

    def fit(self, nchromosome, generation):
        for i in range(nchromosome): # individual
            self.chromosome.append([self.gene()]) # weight have dimension chromosome -> gene, fitness
            out = self.FeedForward(self.chromosome[i][0])
            self.Loss = []
            for j in range(len(self.Fullinput[i])): # sample

                err = out[j] - self.FullTrueOutput[j]

                self.Loss.append(err)

            mse = np.asarray(self.Loss)
            mse = np.power(mse, 2)/ 2
            mse = np.mean(mse) # scalar
            fitness = self.fitness(mse)
            self.chromosome[i].append(fitness)
            print(self.chromosome[i][1])

    def gene(self): # return list
        weight = []
        for  i in range(len(self.Node)):
            if (i == 0):
                # weight.append(2*np.random.rand(len(self.Fullinput), self.Node[i]) - 1)
                weight.append(np.random.uniform(0, 1, (len(self.Fullinput), self.Node[i])))
            else : 
                # weight.append(2*np.random.rand(self.Node[i-1], self.Node[i]) - 1)
                weight.append(np.random.uniform(0, 1, (self.Node[i-1], self.Node[i])))
            
        return weight

    def fitness(self, mse): #### Cal fitness from error | input Integer | how to fitness maximum
        return 1/mse

if __name__ == "__main__":
    main()