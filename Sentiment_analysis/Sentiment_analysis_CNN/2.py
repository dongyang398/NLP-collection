import numpy as np
from random import randint
import tensorflow as tf

maxSeqLength = 250
batchSize = 24
lstmUnits = 128
numClasses = 2
iterations = 100000
numDimensions = 50

wordsList = np.load('./data/wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('./data/wordVectors.npy')
print('Loaded the word vectors!')
print(len(wordsList))
print(wordVectors.shape)
ids = np.load('./data/idsMatrix.npy')


# 标记测试集
def get_test_batch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def test():
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)
    print("%" * 80)
    print(data.shape)
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)

    value, h = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    print(int(value.get_shape()[0]) - 1)

    last = tf.gather(value, int(value.get_shape()[0]) - 1)  # The value of this parameter is 249
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, './models/pretrained_lstm.ckpt-8000')
    iterations = 100
    sum = 0
    for i in range(iterations):
        nextBatch, nextBatchLabels = get_test_batch()
        acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
        sum = sum + acc
    print('------')
    print('Final:%f' % (sum / 100))


if __name__ == '__main__':
    test()