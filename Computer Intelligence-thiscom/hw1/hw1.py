import numpy as np
import csv
import copy

data = []
abc = 1

def Normailize(x):
    mi = x.min()
    ma = x.max()
    x = (x - mi)/(ma - mi)
    return x

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
                self.deltaweight.append(np.ones((len(self.input), self.Node[i])))
                self.deltabias.append(np.ones(self.Node[i]))
                self.rememberweightT_1.append(np.zeros((len(self.input), self.Node[i])))
                self.rememberweightT_2.append(np.zeros((len(self.input), self.Node[i])))
            else:
                self.weight.append(2*np.random.rand(self.Node[i-1], self.Node[i]) - 1)
                self.bias.append(2*np.random.rand(self.Node[i]) - 1)
                self.deltaweight.append(np.ones((self.Node[i-1], self.Node[i])))
                self.deltabias.append(np.ones(self.Node[i]))
                self.rememberweightT_1.append(np.zeros((len(self.input), self.Node[i])))
                self.rememberweightT_2.append(np.zeros((len(self.input), self.Node[i])))

    def FeedForward(self):
        # each sample, each epoch
        self.output = []
        self.output.append(copy.deepcopy(self.input.T))
        # out = np.array(self.output[0])
        # print(out)
        for i in range(len(self.Node)): # feed in each layer
            v = np.dot(copy.deepcopy(np.array(self.output[i])), copy.deepcopy(self.weight[i])) + copy.deepcopy(self.bias[i])
            # print(v)
            out = copy.deepcopy(self.sigmoid(copy.deepcopy(v)))
            self.output.append(copy.deepcopy(out))

    def BackPropagation(self):
        self.gradient = []
        self.err = copy.deepcopy(self.TrueOutput) - copy.deepcopy(self.output[len(self.output)-1])
        self.Loss = []
        Loss = (0.5)*((self.err)**2) # scalar
        #calculate gradient
        self.Loss.append(Loss)
        for i in range(len(self.Node)): # for layer
            self.gradient.append([])
            for j in range(len(self.weight[len(self.Node)-1-i])): # Node in layer
                inpu = copy.deepcopy(self.output[len(self.Node)-i-1].T[j])
                for k in range(len(self.weight[len(self.Node)-1-i][j])): # each weight in layer
                    out = copy.deepcopy(self.output[len(self.Node)-i][k])
                    if ( i == 0 ):
                        gradient = (copy.deepcopy(self.err[k])*copy.deepcopy(self.diffsigmoid(copy.deepcopy(out))))
                        self.gradient[i].append(gradient)
                        bb = copy.deepcopy(self.gradient[i][k])
                        self.deltaweight[len(self.Node)-1-i][j][k] = np.dot(inpu, bb)*self.lr # delta_w scalar in weight 1 line 
                    else :
                        # break
                        gra = np.sum(copy.deepcopy(self.gradient[i-1]))
                        gradient = gra*(copy.deepcopy(self.diffsigmoid(out)))*copy.deepcopy(self.weight[len(self.Node)-i-1][j][k])
                        self.gradient[i].append(copy.deepcopy(gradient))
                        self.deltaweight[len(self.Node)-1-i][j][k] = inpu*copy.deepcopy(self.gradient[i][k])*self.lr # delta_w scalar in weight 1 line 

                    if ( self.countsample <= 1 ):
                        www = copy.deepcopy(self.weight[len(self.Node)-1-i][j][k])
                        dw = copy.deepcopy(self.deltaweight[len(self.Node)-1-i][j][k])
                        self.weight[len(self.Node)-1-i][j][k] = www + dw # scalar
                    else : # with momentum
                        a = copy.deepcopy(self.rememberweightT_1[len(self.Node)-1-i][j][k]) - copy.deepcopy(self.rememberweightT_2[len(self.Node)-1-i][j][k])
                        www = copy.deepcopy(self.weight[len(self.Node)-1-i][j][k])
                        dw = copy.deepcopy(self.deltaweight[len(self.Node)-1-i][j][k])
                        self.weight[len(self.Node)-1-i][j][k] = www + self.momentum*(a) + dw # scalar
        self.rememberweightT_2 = copy.deepcopy(self.rememberweightT_1)
        self.rememberweightT_1 = copy.deepcopy(self.weight)

    def fit(self, epoch, Learningrate, Momentum):
        self.lr = Learningrate
        self.momentum = Momentum
        self.countsample = 0
        for i in range(epoch): # epoch
            for j in range(len(self.Fullinput)): # sample
                self.input = self.Fullinput[j]
                self.TrueOutput = self.FullTrueOutput[j]
                if ( i == 0 and j == 0):
                    self.createweight()    
                
                self.FeedForward()
                self.BackPropagation()

                self.countsample += 1
                # break
            sumloss = np.array(self.Loss).mean()
            # print("Epoch = " + str(i+1) + "/" + str(epoch) + ", Loss =", sumloss)
    
            self.minimumLoss = min(sumloss, self.minimumLoss)
    
    def predict_data(self, inpu, output):
        self.Fullinput = inpu
        self.FullTrueOutput = output
        self.Loss = []
        for j in range(len(self.Fullinput)):
            self.input = self.Fullinput[j]
            self.TrueOutput = self.FullTrueOutput[j]

            self.FeedForward()

            self.err = self.TrueOutput - self.output[len(self.output)-1]
            Loss = (0.5)*((self.err)**2) # scalar
            self.Loss.append(Loss)

        MeanLoss = np.array(self.Loss).mean()
        return(MeanLoss)

    def confusion_matrix(self, inpu, output):
        self.Fullinput = copy.deepcopy(inpu)
        self.FullTrueOutput = copy.deepcopy(output)
        self.Loss = []
        sizeclass = len(output[0])
        classconfusion = np.zeros((sizeclass, sizeclass))
        for j in range(len(self.Fullinput)):
            self.input = self.Fullinput[j]
            self.TrueOutput = self.FullTrueOutput[j]

            self.FeedForward()
            self.err = self.TrueOutput - self.output[len(self.output)-1]
            classconfusion[self.TrueOutput.argmax()][self.output[len(self.output)-1].argmax()] += 1
            # print(self.output)
        print()
        print("          Predict")
        print("       |-----------")
        for i in range(sizeclass):
            if ( i == 0 ) :
                print("Actual |", classconfusion[i])
            else :
                print("       |", classconfusion[i])
        accuracy = 0
        for i in range(len(classconfusion)):
            accuracy += classconfusion[i][i]
        print()
        print("Accuracy = " + str((float(accuracy)/len(output))*100) + ' %, ' + str(int(accuracy)) + "/" + str(len(output)))

# first 
if ( abc == 1 ):
    with open('dataset.csv', 'rt')as f:
        d = csv.reader(f)
        for row in d:
            data.append(row)

    x_train = []
    y_true = []

    for i in range(len(data)):
        if ( i > 1 ):
            inpu = []
            for j in range(len(data[i])):
                if ( j < 8 ):
                    inpu.append(float(data[i][j]))
                else:
                    y_true.append(float(data[i][j]))
            x_train.append(inpu)
    
    fold = 10
    crossvalidation = int(len(x_train)*fold/100)

    for i in range(fold):
        if i == (fold-1) :
            x_train_testingset = x_train[0+i*crossvalidation:len(x_train)]
            y_true_testingset = y_true[0+i*crossvalidation:len(y_true)]

            x_train_trainingset = x_train[0:0+i*crossvalidation]
            y_true_trainingset = y_true[0:0+i*crossvalidation]
        
        else:
            x_train_testingset = x_train[0+i*crossvalidation:crossvalidation+i*crossvalidation]
            y_true_testingset = y_true[0+i*crossvalidation:crossvalidation+i*crossvalidation]

            x_train_trainingset1 = x_train[0:i*crossvalidation]
            x_train_trainingset2 = x_train[crossvalidation*(i+1):len(x_train)]

            x_train_trainingset = x_train_trainingset1 + x_train_trainingset2

            y_true_trainingset1 = y_true[0:i*crossvalidation]
            y_true_trainingset2 = y_true[crossvalidation*(i+1):len(x_train)]

            y_true_trainingset = y_true_trainingset1 + y_true_trainingset2
        
        x_train_trainingset = np.array(x_train_trainingset)
        y_true_trainingset = np.array(y_true_trainingset)

        x_train_trainingset = Normailize(x_train_trainingset)
        y_true_trainingset = Normailize(y_true_trainingset)

        nn = NeuralNetwork(x_train_trainingset, y_true_trainingset)

        nn.addLayer(3)
        nn.addLayer(3)
        nn.addLayer(2)
        nn.addLayer(2)
        nn.addLayer(1)

        x_train_testingset = np.array(x_train_testingset)
        y_true_testingset = np.array(y_true_testingset) 

        x_train_testingset = Normailize(x_train_testingset)
        y_true_testingset = Normailize(y_true_testingset)

        epoch = 100

        print("Fold " + str(i+1) + ", Epoch =", epoch, "epochs")
        # nn.fit(epoch, 0.00173, 0)
        nn.fit(epoch, 0.00375, 0.5)
        print("MSE =", nn.predict_data(x_train_testingset, y_true_testingset))
        print("------------------------------------------")

# second
elif ( abc == 2 ):
    cou = 0
    with open('cross.pat', 'rt')as f:
        d = csv.reader(f)
        for row in d:
            if ( cou % 3 != 0 ):
                data.append(row)
            cou += 1
    x_train = []
    y_true = []
    for i in range(len(data)):
        data[i][0] = data[i][0].split()
        inpu = []
        for row in data[i][0]:
            inpu.append(float(row))
        if ( i % 2 == 0 ):
            x_train.append(inpu)
        else:
            y_true.append(inpu)
    
    fold = 10
    crossvalidation = int(len(x_train)*fold/100)

    for i in range(fold):
        if i == (fold-1) :
            x_train_testingset = x_train[0+i*crossvalidation:len(x_train)]
            y_true_testingset = y_true[0+i*crossvalidation:len(y_true)]

            x_train_trainingset = x_train[0:0+i*crossvalidation]
            y_true_trainingset = y_true[0:0+i*crossvalidation]
        
        else:
            x_train_testingset = x_train[0+i*crossvalidation:crossvalidation+i*crossvalidation]
            y_true_testingset = y_true[0+i*crossvalidation:crossvalidation+i*crossvalidation]

            x_train_trainingset1 = x_train[0:i*crossvalidation]
            x_train_trainingset2 = x_train[crossvalidation*(i+1):len(x_train)]

            x_train_trainingset = x_train_trainingset1 + x_train_trainingset2

            y_true_trainingset1 = y_true[0:i*crossvalidation]
            y_true_trainingset2 = y_true[crossvalidation*(i+1):len(x_train)]

            y_true_trainingset = y_true_trainingset1 + y_true_trainingset2
        
        x_train_trainingset = np.array(x_train_trainingset)
        y_true_trainingset = np.array(y_true_trainingset)
        
        # x_train_trainingset = Normailize(x_train_trainingset)
        # y_true_trainingset = Normailize(y_true_trainingset)

        for j in range(len(y_true_trainingset)):
            if ( y_true_trainingset[j][0] == 0 ):
                y_true_trainingset[j][0] = 0.1
            if ( y_true_trainingset[j][1] == 0 ):
                y_true_trainingset[j][1] = 0.1
            if ( y_true_trainingset[j][0] == 1 ):
                y_true_trainingset[j][0] = 0.9
            if ( y_true_trainingset[j][1] == 1 ):
                y_true_trainingset[j][1] = 0.9

        x_train_testingset = np.array(x_train_testingset)
        y_true_testingset = np.array(y_true_testingset)

        nn = NeuralNetwork(x_train_trainingset, y_true_trainingset)
  
        # x_train_testingset = Normailize(x_train_testingset)
        # y_true_testingset = Normailize(y_true_testingset)

        for j in range(len(y_true_testingset)):
            if ( y_true_testingset[j][0] == 0 ):
                y_true_testingset[j][0] = 0.1
            if ( y_true_testingset[j][1] == 0 ):
                y_true_testingset[j][1] = 0.1
            if ( y_true_testingset[j][0] == 1 ):
                y_true_testingset[j][0] = 0.9
            if ( y_true_testingset[j][1] == 1 ):
                y_true_testingset[j][1] = 0.9

        # nn.addLayer(8)
        nn.addLayer(5)
        # nn.addLayer(8)
        # nn.addLayer(3)
        nn.addLayer(2)
        nn.addLayer(1)

        epoch = 100

        print("Fold " + str(i+1) + ", Epoch =", epoch, "epochs")
        nn.fit(epoch, 0.001, 0.5)
        # break
        nn.confusion_matrix(x_train_testingset, y_true_testingset)
        print("------------------------------------------")
        # break
        