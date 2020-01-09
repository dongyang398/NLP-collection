#coding:utf-8


import os
import numpy as np
import tensorflow as tf
import input_text_data
import CNN_model


N_CLASSES = 2
IMG_H = 250
IMG_W = 50
# BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 0.0001
maxSeqLength=250
numDimensions=50
wordVectors = np.load('./data/wordVectors.npy')
logs_train_dir='./text_log'


def training():
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    labels = tf.placeholder(tf.int32, [batchSize])

    data = tf.nn.embedding_lookup(wordVectors, input_data)
    data = tf.reshape(data, [batchSize, maxSeqLength, numDimensions, 1])

    train_logits = CNN_model.inference1(data, batchSize, N_CLASSES)
    train_loss = CNN_model.losses(train_logits, labels)
    train_op = CNN_model.trainning(train_loss, learning_rate)
    train_acc = CNN_model.evaluation(train_logits, labels)

    sess = tf.InteractiveSession()
    tf.summary.scalar('Loss', train_loss)
    tf.summary.scalar('Accuracy', train_acc)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        for i in range(MAX_STEP):
            train_batch, train_label_batch = input_text_data.getTrainBatch()
            _, tra_loss, tra_acc =sess.run([train_op, train_loss, train_acc],{input_data: train_batch,labels:train_label_batch})

            # 汇总到Tensorboard
            if i % 1000 == 0:
                print("Step %d, train loss = %.2f, train accuracy = %.2f%%" % (i, tra_loss, tra_acc))

                test_batch, test_label_batch = input_text_data.getTestBatch()
                summary = sess.run(merged, {input_data: test_batch , labels: test_label_batch})
                writer.add_summary(summary, i)
                test_loss, test_acc = sess.run([train_loss, train_acc],
                                         {input_data: test_batch, labels: test_label_batch})
                print("*************, test loss = %.2f, test accuracy = %.2f%%" % (test_loss, test_acc))

            if i % 2000 == 0 or (i + 1) == MAX_STEP:
                save_path = saver.save(sess, "./models/pretrained_CNN.ckpt", global_step=i)
                print("saved to %s" % save_path)
        writer.close()


def test():
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    labels = tf.placeholder(tf.int32, [batchSize])

    data = tf.nn.embedding_lookup(wordVectors, input_data)
    data = tf.reshape(data, [batchSize, maxSeqLength, numDimensions, 1])

    train_logits = CNN_model.inference1(data, batchSize, N_CLASSES)
    train_loss = CNN_model.losses(train_logits, labels)
    train_acc = CNN_model.evaluation(train_logits, labels)

    all_accuracy = 0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './models/pretrained_CNN.ckpt-20000')

        for i in range(2000):
            train_batch, train_label_batch = input_text_data.get_allTest(i)
            tra_loss, tra_acc = sess.run([train_loss, train_acc],
                                         {input_data: train_batch, labels: train_label_batch})
            all_accuracy = all_accuracy+tra_acc
        print('All_accuracy:')
        print(all_accuracy/2000)


if __name__ == '__main__':
    training()
    test()