from random import randint
import tensorflow as tf
import numpy as np

batch_size = 24
lstm_units = 64
num_labels = 2
iterations = 100
lr = 0.001
max_seq_num = 250  # 时间序列长度
ids = np.load('idsMatrix.npy')

# 获取训练集
def get_train_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


# 标记测试集
def get_test_batch():
    labels = []
    arr = np.zeros([batch_size, max_seq_num])
    for i in range(batch_size):
        num = randint(11499, 13499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num - 1:num]
    return arr, labels


def train():
    tf.reset_default_graph()   # 重置全局默认图形
    labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num])
    data = tf.Variable(tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data)
    # 定义一层lstm_cell
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstm_units)
    # 添加 dropout layer
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.8)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.exists("models") and os.path.exists("models/checkpoint"):
            saver.restore(sess, tf.train.latest_checkpoint('models'))
        else:
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
        iterations = 100
        for step in range(iterations):
            next_batch, next_batch_labels = get_test_batch()
            if step % 20 == 0:
                print("step:", step, " 正确率:", (sess.run(
                    accuracy, {input_data: next_batch, labels: next_batch_labels})) * 100)

        if not os.path.exists("models"):
            os.mkdir("models")
        save_path = saver.save(sess, "models/model.ckpt")
        print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
    train()
