import time
import pandas as pd
import util.network as network
import util.input_data as input_data
import numpy as np
import tensorflow as tf
import util.util as ul
import util.network as net
import util.common as common

CHANNELS = 3
IMAGE_SIZE = 32
BATCH_SIZE = 64
LEARNING_RATE = 0.005
DISPLAY_IMAGE_INDEX = 10
CLASS_NUM = 10

train_images, train_labels, test_images, test_labels = input_data.get_cifar_data()


def get_image_normalization_matrix(images):
    num = len(images)

    img_data = np.ndarray(shape=(num, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)

    for index in range(num):
        data = np.array(np.reshape(images[index], newshape=[IMAGE_SIZE, IMAGE_SIZE, CHANNELS]), dtype=np.float32)

        data[:, :, 0] = data[:, :, 0] / 255.0
        data[:, :, 1] = data[:, :, 1] / 255.0
        data[:, :, 2] = data[:, :, 2] / 255.0

        img_data[index] = data

    return img_data


train_images = get_image_normalization_matrix(train_images)
test_images = get_image_normalization_matrix(test_images)
train_labels = pd.get_dummies(train_labels).values
test_labels = pd.get_dummies(test_labels).values

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)


def shuffle(images, labels):
    order = np.random.permutation(images.shape[0])
    labels = labels[order, :]
    images = images[order, :, :, :]

    return images, labels


train_images, train_labels = shuffle(train_images, train_labels)

start = -BATCH_SIZE


def next_batch(batch_size):
    global start
    global train_labels
    global train_images

    start += batch_size
    if start + batch_size > train_labels.shape[0]:
        train_images, train_labels = shuffle(train_images, train_labels)
        start = 0

    end = start + batch_size

    return train_images[start:end, :, :, :], train_labels[start:end, :]


def VGG_16(inputs, keep_prob):
    with tf.name_scope('reshape'):
        inputs = tf.reshape(inputs, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('layer1'):
        inputs = net.conv2_layer(inputs, 3, 64, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 64, 1, padding='SAME')
        inputs = net.max_pool_layer(inputs, 2, 2, padding='VALID')

    with tf.name_scope('layer2'):
        inputs = net.conv2_layer(inputs, 3, 128, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 128, 1, padding='SAME')
        inputs = net.max_pool_layer(inputs, 2, 2, padding='VALID')

    with tf.name_scope('layer3'):
        inputs = net.conv2_layer(inputs, 3, 256, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 256, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 256, 1, padding='SAME')
        inputs = net.max_pool_layer(inputs, 2, 2, padding='VALID')

    with tf.name_scope('layer4'):
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.max_pool_layer(inputs, 2, 2, padding='VALID')

    with tf.name_scope('layer5'):
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.conv2_layer(inputs, 3, 512, 1, padding='SAME')
        inputs = net.max_pool_layer(inputs, 2, 2, padding='VALID')

    with tf.name_scope('flat'):
        shape = inputs.shape
        inputs = tf.reshape(inputs, [-1, shape[1].value * shape[2].value * shape[3].value])

    with tf.name_scope('layer6'):
        inputs = net.fully_connection_layer(inputs, 4096, relu=True, keep_prob=keep_prob)
        inputs = net.fully_connection_layer(inputs, 4096, relu=True, keep_prob=keep_prob)
        inputs = net.fully_connection_layer(inputs, CLASS_NUM, relu=False, keep_prob=1.0)

    # with slim.arg_scope([slim.conv2d, slim.fully_connected],
    #                     activation_fn=tf.nn.relu,
    #                     weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
    #                     weights_regularizer=slim.l2_regularizer(0.0005)):
    #     inputs = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    #     inputs = slim.max_pool2d(inputs, [2, 2], scope='pool1')
    #     inputs = slim.repeat(inputs, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    #     inputs = slim.max_pool2d(inputs, [2, 2], scope='pool2')
    #     inputs = slim.repeat(inputs, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    #     inputs = slim.max_pool2d(inputs, [2, 2], scope='pool3')
    #     inputs = slim.repeat(inputs, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    #     inputs = slim.max_pool2d(inputs, [2, 2], scope='pool4')
    #     inputs = slim.repeat(inputs, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    #     inputs = slim.max_pool2d(inputs, [2, 2], scope='pool5')
    #     inputs = slim.fully_connected(inputs, 4096, scope='fc6')
    #     inputs = slim.dropout(inputs, keep_prob, scope='dropout6')
    #     inputs = slim.fully_connected(inputs, 4096, scope='fc7')
    #     inputs = slim.dropout(inputs, keep_prob, scope='dropout7')
    #     inputs = slim.fully_connected(inputs, 1000, activation_fn=None, scope='fc8')

    return inputs


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM])

hat = VGG_16(x, keep_prob)

y = tf.nn.softmax(hat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), dtype=tf.float32))

EPOCHS = 500

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start_time = common.get_clock()
    for step in range(1, EPOCHS + 1):
        xs, ys = next_batch(BATCH_SIZE)

        feed_dict = {x: xs, y_: ys, keep_prob: 0.5}
        sess.run(train_step, feed_dict=feed_dict)

        if step % 20 == 0:
            xs, ys = next_batch(BATCH_SIZE)

            feed_dict = {x: xs, y_: ys, keep_prob: 0.75}
            valid_accuracy = accuracy.eval(feed_dict=feed_dict)

            print('step {} valid accuracy {}'.format(step, valid_accuracy))

    end_time = common.get_clock()
    common.get_format_time(start_time, end_time)

    xs, ys = next_batch(BATCH_SIZE)
    feed_dict = {x: xs, y_: ys, keep_prob: 1.0}
    test_accuracy = accuracy.eval(feed_dict=feed_dict)

    print('test accuracy {}'.format(test_accuracy))
