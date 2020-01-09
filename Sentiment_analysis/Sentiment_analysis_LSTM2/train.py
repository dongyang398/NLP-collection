import numpy as np
from random import randint
import tensorflow as tf
import datetime

batchSize = 24
lstmUnits1 = 128
lstmUnits2 = 128
numClasses = 2
numDimensions = 50  # 每个词向量的维数
iterations = 10000
maxSeqLength = 250
wordVectors = np.load('./data/wordVectors.npy')
print('Loaded the word vectors!')
print(wordVectors.shape)
ids = np.load('./data/idsMatrix.npy')

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels


def train():
    # 重置计算图
    tf.reset_default_graph()
    # 定义占位符
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data) #

    # 定义 LSTM
    lstmCell1 = tf.contrib.rnn.BasicLSTMCell(lstmUnits1)
    lstmCell1 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell1, output_keep_prob=0.5)

    lstmCell2 = tf.contrib.rnn.BasicLSTMCell(lstmUnits2)
    lstmCell2 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell2, output_keep_prob=0.5)
    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstmCell1, lstmCell2], state_is_tuple=True)
    value, _ = tf.nn.dynamic_rnn(mlstm_cell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits1, numClasses]))   # 正态随机生成 W
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))  # bias 初始化为0.1
    value = tf.transpose(value, [1, 0, 2])  # 转秩
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))  # 平均正确率

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 汇总到tensorboard
    sess = tf.InteractiveSession()
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(iterations):
            # 喂数据
            nextBatch, nextBatchLabels = getTrainBatch();
            sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

            # 汇总到Tensorboard
            if (i % 1000 == 0):
                summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
                writer.add_summary(summary, i)
                acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
                print('training! Step%d Training accuracy:%f' % (i, acc))

            if (i % 1000 == 0 and i != 0):
                save_path = saver.save(sess, "./models/pretrained_lstm.ckpt", global_step=i)
                print("saved to %s" % save_path)
        writer.close()

if __name__ == '__main__':
    train()