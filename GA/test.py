import numpy as np
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
#### data | 1). ID Number, 2). Diagnosis (M = malignant, B = benign) -> Class 3). 3 -> 32 is features ####

def main():
    data = getData()

    data = normalizeList(data)

    input_x = data[1:] # 2 dim for each features and sample
    true_x = data[0] # 1 dim for each sample

    input_x = transposeList(input_x)

    x_train, y_train, x_test, y_test= createFold(10, input_x, true_x)

    print((len(x_train), len(x_test)))
    print((len(x_train[0]), len(x_test)))
    print((len(x_train[0][0]), len(x_test)))

    for i in range(len(x_train)): #### Fold
        print("Fold {}" .format(i+1))
        x_train[i] = transposeList(x_train[i])
        x_test[i] = transposeList(x_test[i])

        neural = NeuralNetwork(x_train[i], y_train[i])
        neural.addLayer(50)
        # neural.addLayer(10)
        # neural.addLayer(5)
        # neural.addLayer(10)
        # neural.addLayer(5)
        neural.addLayer(1)
        
        neural.fit(20, 35)
        # neural.plotFitness(i+1)
        neural.evaluate(x_test[i], y_test[i])

        print("---------------")
        break

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
                new[i].append(0.1)
            elif listdata[i][j] == 'B' :
                new[i].append(0.9)
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
        self.chromosome = {}
        self.nchromosome = 0
        self.child = []
        self.fitnessplot = []

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
            v = np.dot(copy.deepcopy(self.output[i]), copy.deepcopy(chromosome[i]))
            out = self.sigmoid(v)
            self.output.append(out)
        
        return out

    def fit(self, nchromosome, generation):
        self.nchromosome = nchromosome
        self.fitnessplot = []
        for k in range(generation) :
            fitne = []
            self.chromosome = {}
            for i in range(nchromosome): # individual
                if k == 0 :
                    temp = self.gene()
                    temp = np.asarray(temp)
                    out = self.FeedForward(temp)
                else :
                    out = self.FeedForward(self.child[i])

                self.Loss = []

                for j in range(len(self.Fullinput[0])): # sample
                    err = out[j] - self.FullTrueOutput[j]
                    self.Loss.append(err)

                mse = np.asarray(copy.deepcopy(self.Loss))
                mse = np.power(mse, 2)/ 2
                mse = np.mean(mse) # scalar
                fitness = self.fitness(mse)
                fitne.append(fitness)
                if k == 0 :
                    self.chromosome[fitness] = temp
                else :
                    self.chromosome[fitness] = self.child[i]
            
            fitne = np.asarray(copy.deepcopy(fitne))
            fitne = np.mean(fitne)
            # print("Generation {} Fitness {}" .format(k+1, fitne))
            self.fitnessplot.append(fitne)
            self.chromosome = self.matingPool()
            n_crossingpoint = np.random.randint(0, nchromosome-1)
            child = self.crossover(n_crossingpoint)
            self.child = self.mutate(child)

    def gene(self): # return list
        weight = []
        for  i in range(len(self.Node)):
            if (i == 0):
                weight.append(np.random.uniform(-1, 1, (len(self.Fullinput), self.Node[i])))
            else : 
                weight.append(np.random.uniform(-1, 1, (self.Node[i-1], self.Node[i])))
            
        return weight

    def fitness(self, mse): #### Cal fitness from error | input Integer | how to fitness maximum
        return 1/mse

    def matingPool(self): # sort fitness value in chromosome
        matingpool = {}
        cou = 0
        for i in sorted(self.chromosome.keys(), reverse=True) :

            matingpool[i] = copy.deepcopy(self.chromosome[i])
            cou += 1

        return matingpool

    def crossover(self, nchild): #### return gene without fitness
        child = [] # have only gene
        for i in range(int(nchild/2)) :
            a = random.choice(list(self.chromosome.keys()))
            b = random.choice(list(self.chromosome.keys()))
            
            genea = []
            geneb = []
            for k in range(len(self.chromosome[a])):

                ga = np.asarray(self.chromosome[a][k])
                gb = np.asarray(self.chromosome[b][k])

                shape_gene = ga.shape
                genea_flatten = ga.flatten()
                geneb_flatten = gb.flatten()

                ncrossingpoint = np.random.randint(0, genea_flatten.shape[0])

                list_crossingpoint = random.sample(range(genea_flatten.shape[0]), ncrossingpoint)

                for j in list_crossingpoint :
                    genea_flatten[j], geneb_flatten[j] = geneb_flatten[j], genea_flatten[j]

                ga = np.reshape(genea_flatten, shape_gene)
                gb = np.reshape(geneb_flatten, shape_gene)

                genea.append(ga)
                geneb.append(gb)

            child.append(genea)
            child.append(geneb)

        if len(child) != self.nchromosome :
            n_residual = self.nchromosome - len(child)
            cou = 0
            for i in self.chromosome.keys() :
                if cou == n_residual :
                    break
                child.append(copy.deepcopy(self.chromosome[i]))
                cou += 1

        return child

    def mutate(self, child):
        child = copy.deepcopy(child)
        n_chromosome = np.random.randint(len(child)) ### random n chromosome to mutate
        choice_chromosome = random.sample(range(len(child)), n_chromosome) # random choice n chromosome to mutate
        for i in range(len(choice_chromosome)):
            for k in range(len(child[i])):
                target = child[i][k].shape
                gene = child[i][k].flatten()
                n_gene = np.random.randint(gene.shape[0])
                list_genemutate = random.sample(range(gene.shape[0]), n_gene)
                list_mutate = np.random.uniform(-1, 1, n_gene)

                for j in range(n_gene):
                    gene[list_genemutate[j]] += copy.deepcopy(list_mutate[j])
                
                child[i][k] = np.reshape(gene, target)
        return child

    def evaluate(self, x_test, y_test):
        x_test = np.asarray(copy.deepcopy(x_test))
        y_test = np.asarray(copy.deepcopy(y_test))

        self.input = x_test
        fitne = []
        for i in range(len(self.child)):

            predict = self.FeedForward(self.child[i])

            Loss = []
            for j in range(len(y_test)):
                err = predict[j] - y_test[j]
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
        ax.set(xlabel='Generation', ylabel='Fitness', title='Fold {}' .format(fold))
        fig.savefig("Fold {}.png" .format(fold))

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