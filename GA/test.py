import numpy as np
import copy
#### data | 1). ID Number, 2). Diagnosis (M = malignant, B = benign) -> Class 3). 3 -> 32 is features ####

def main():
    data = getData()
    print(data)

def getData(): ### return list
    path = 'wdbc.data.txt'
    listdata = []
    with open(path, 'r') as fi:
        row = [line for line in fi.read().splitlines()]

    for r in row :
        listdata.append(r.split(','))

    return listdata

class NeuralNetwork:
    def __init__(self, inpu, out):
        # np.random.seed(1)
        self.Fullinput = inpu
        self.FullTrueOutput = out
        self.input = None
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

    def FeedForward(self, weight):
        # each sample, each generation
        self.output = []
        self.output.append(self.input.T)
        out = np.array(self.output[0])
        # print(out)
        for i in range(len(self.Node)): # feed in each layer
            v = np.dot(self.output[i], self.weight[i]) + self.bias[i]
            # print(v)
            out = self.sigmoid(v)
            self.output.append(out)
        
        return out

    def fit(self, nchromosome, generation):
        for i in range(nchromosome): # individual
            self.weight.append(self.gene()) # weight have dimension chromosome -> gene
            for j in range(len(self.Fullinput)): # sample
                
                out = self.FeedForward(self.weight[i])
                
                err = out - self.FullTrueOutput[j]

                self.Loss.append(err)
                # break
            mse = np.asarray(self.Loss).mean()

            # self.minimumLoss = min(sumloss, self.minimumLoss)

    def gene(self): # return list
        weight = []
        for  i in range(len(self.Node)):
            if (i == 0):
                weight.append(2*np.random.rand(len(self.input), self.Node[i]) - 1)
            else : 
                weight.append(2*np.random.rand(self.Node[i-1], self.Node[i]) - 1)
            
        return weight

    def fitness(self, mse): #### Cal fitness from error | input Integer
        return
        # print("A")

if __name__ == "__main__":
    main()