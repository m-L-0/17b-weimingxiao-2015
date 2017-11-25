# from utils import mnist_reader
from numpy import *  
import operator
import read  
#     X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
#     X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
#     X_test=X_test[:100];y_test=y_test[:100]
# from tensorflow.examples.tutorials.mnist import input_data
# X_train, y_train =input_data.read_data_sets('data/fashion').train.next_batch(50000)
# X_test, y_test = input_data.read_data_sets('data/fashion').test.next_batch(200)

[X_train, y_train] = read.read_and_decode('/home/r/fashion-mnist/train.tfrecords',50000)
[X_test, y_test] = read.read_and_decode('/home/r/fashion-mnist/test.tfrecords',200)
def change(X):
    s,d,f=X.shape
    b=empty((s,d*f))
    for i in range(s):
        c=X[i]
        b[i]=c.reshape(d*f)
    X=b
    return X
X_train=change(X_train)
X_test=change(X_test)
# create a dataset which contains 4 samples with 2 classes  
def createDataSet():  
    # create a matrix: each row as a sample  
    group = array(X_train)  
    labels = y_train # four samples and two classes  
    return group, labels  
    
# classify using kNN  
def kNNClassify(newInput, dataSet, labels, k):
    
    numSamples = dataSet.shape[0] # shape[0] stands for the num of row  行数
    
    ## step 1: calculate Euclidean distance  计算欧式距离
    # tile(A, reps): Construct an array by repeating A reps times  
    # the following copy numSamples rows for dataSet  
    
    diff = tile(newInput, (numSamples, 1)) - dataSet # Subtract element-wise  
    squaredDiff = diff ** 2 # squared for the subtract  
    squaredDist = sum(squaredDiff, axis = 1) # sum is performed by row  
    distance = squaredDist ** 0.5  
    
    ## step 2: sort the distance  距离分类
    # argsort() returns the indices that would sort an array in a ascending order  
    sortedDistIndices = argsort(distance)  
    
    classCount = {} # define a dictionary (can be append element)  
    for i in range(k):  
        ## step 3: choose the min k distance  距离最小的几个值
        voteLabel = labels[sortedDistIndices[i]]  
    
        ## step 4: count the times labels occur  标签出现次数
        # when the key voteLabel is not in dictionary classCount, get()  
        # will return 0  
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1  
    
    ## step 5: the max voted class will return  
    maxCount = 0  
    for key, value in classCount.items():  
        if value > maxCount:  
            maxCount = value  
            maxIndex = key  
        if value == 1:
            maxIndex = labels[sortedDistIndices[0]] 
    return maxIndex
dataSet, labels = createDataSet()  
s = 0
print('start the Knn')
for i in range(len(X_test)):
    testX = array(X_test[i])  
    outputLabel = kNNClassify(testX, dataSet, labels, 3)  
    if outputLabel == y_test[i]:
        s = s + 1
w = s/len(y_test)
print(w)