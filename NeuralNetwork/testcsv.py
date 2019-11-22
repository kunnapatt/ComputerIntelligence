import csv
data = []
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
                inpu.append(data[i][j])
            else:
                y_true.append(data[i][j])
                
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


    # print(len(x_train_trainingset))
    # print(len(y_true_trainingset))
    # print(len(x_train_testingset))
    # print(len(y_true_testingset))


# print(y_true)

