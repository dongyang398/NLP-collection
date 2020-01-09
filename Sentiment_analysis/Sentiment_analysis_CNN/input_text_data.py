import numpy as np
from random import randint

maxSeqLength = 250
batchSize = 32
numDimensions=50

wordVectors = np.load('./data/wordVectors.npy')
print ('Loaded the word vectors!')
ids = np.load('./data/idsMatrix.npy')

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append(0)
        else:
            num = randint(13499,24999)
            labels.append(1)
        arr[i] = ids[num-1:num]

    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append(0)
        else:
            labels.append(1)
        arr[i] = ids[num - 1:num]
    return arr, labels

def get_allTest(step):
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = 11499+1+step
        if (num <= 12499):
            labels.append(0)
        else:
            labels.append(1)
        arr[i] = ids[num - 1:num]
    return arr, labels
