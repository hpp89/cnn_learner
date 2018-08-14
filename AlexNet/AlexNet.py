import cv2
import numpy as np
import util.input_data as input_data
import tensorflow as tf

CHANNELS = 3
IMAGE_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DISPLAY_IMAGE_INDEX = 10


data = input_data.get_ImageNet()

def next_batch(batch_size):
    pass


def weight_variable(shape):
    initial = tf.truncated_normal(dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2_layer(x, filter_size, out_num, stride_size, padding='SAME'):
    in_num = x.shape[2].value

    weight = weight_variable([filter_size, filter_size, in_num, out_num])
    bias = bias_variable([out_num])

    conv = tf.nn.conv2d(x, weight, strides=[1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def max_pool_layer(x, k_size, stride_size, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)


def fully_connection_layer(x, out_num, relu=False, keep_prob=1.0):
    in_num = x.shape[2].value

    weight = weight_variable([in_num, out_num])
    bias = bias_variable([out_num])

    fc = tf.add(tf.matmul(x, weight), bias)

    if relu:
        fc = tf.nn.relu(fc)

    if keep_prob != 1.0:
        fc = tf.nn.dropout(fc, keep_prob=keep_prob)

    return fc


def AlexNet(x, keep_prob):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('layer1'):
        conv1 = conv2_layer(x, 11, 96, 4, padding='VALID')
        pool1 = max_pool_layer(conv1, 3, 2, padding='VALID')

    with tf.name_scope('layer2'):
        conv2 = conv2_layer(pool1, 5, 256, 1, padding='SAME')
        pool2 = max_pool_layer(conv2, 3, 2, padding='VALID')

    with tf.name_scope('layer3'):
        conv3 = conv2_layer(pool2, 3, 384, 1, padding='SAME')

    with tf.name_scope('layer4'):
        conv4 = conv2_layer(conv3, 3, 384, 1, padding='SAME')

    with tf.name_scope('layer5'):
        conv5 = conv2_layer(conv4, 3, 256, 1, padding='SAME')
        pool5 = max_pool_layer(conv5, 3, 2, padding='VALID')

    with tf.name_scope('flat'):
        shape = pool5.shape
        pool5_flat = tf.reshape(pool5, [-1, shape[1].value * shape[2].value * shape[3].value])

    with tf.name_scope('layer6'):
        fc1 = fully_connection_layer(pool5_flat, 4096, relu=True, keep_prob=keep_prob)

    with tf.name_scope('layer7'):
        fc2 = fully_connection_layer(fc1, 4096, relu=True, keep_prob=keep_prob)

    with tf.name_scope('layer8'):
        fc3 = fully_connection_layer(fc2, 1000, relu=False, keep_prob=.0)

    return fc3


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
y_ = tf.placeholder(tf.float32, [None, 1000])

logits = AlexNet(x, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

EPOCHS = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1, EPOCHS + 1):
        xs, ys = next_batch(BATCH_SIZE)

        feed_dict = {x: xs, y_: ys, keep_prob: 0.5}
        _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)

        if step % 20 == 0:
            xs, ys = next_batch(BATCH_SIZE)

            feed_dict = {x: xs, y_: ys, keep_prob: 0.75}
            valid_accuracy = accuracy.eval(feed_dict=feed_dict)

            print('step {} valid accuracy {}, loss {}'.format(step, valid_accuracy, train_loss))


    xs, ys = next_batch(BATCH_SIZE)
    feed_dict = {x: xs, y_: ys, keep_prob: 1.0}
    test_accuracy = accuracy.eval(feed_dict=feed_dict)

    print('test accuracy {}'.format(test_accuracy))
