import time
import pandas as pd
import util.network as network
import util.input_data as input_data
import numpy as np
import tensorflow as tf
import util.util as ul
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


def AlexNet(x, keep_prob):
    # with tf.name_scope('reshape'):
    #     x = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('layer1'):
        conv1 = network.conv2_layer(x, 3, 32, 1, padding='SAME')
        pool1 = network.max_pool_layer(conv1, 2, 2, padding='VALID')

    with tf.name_scope('layer2'):
        conv2 = network.conv2_layer(pool1, 3, 64, 1, padding='SAME')
        pool2 = network.max_pool_layer(conv2, 2, 2, padding='VALID')

    with tf.name_scope('layer3'):
        conv3 = network.conv2_layer(pool2, 3, 128, 1, padding='SAME')

    with tf.name_scope('layer4'):
        conv4 = network.conv2_layer(conv3, 3, 256, 1, padding='SAME')

    with tf.name_scope('layer5'):
        conv5 = network.conv2_layer(conv4, 3, 512, 1, padding='SAME')
        pool5 = network.max_pool_layer(conv5, 2, 2, padding='VALID')

    with tf.name_scope('flat'):
        shape = pool5.shape
        pool5_flat = tf.reshape(pool5, [-1, shape[1].value * shape[2].value * shape[3].value])

    with tf.name_scope('layer6'):
        fc1 = network.fully_connection_layer(pool5_flat, 2048, relu=True, keep_prob=keep_prob)

    with tf.name_scope('layer7'):
        fc2 = network.fully_connection_layer(fc1, 2048, relu=True, keep_prob=keep_prob)

    with tf.name_scope('layer8'):
        fc3 = network.fully_connection_layer(fc2, CLASS_NUM, relu=False, keep_prob=1.0)

    return fc3


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
y_ = tf.placeholder(tf.float32, [None, CLASS_NUM])

logits = AlexNet(x, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

EPOCHS = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    start_time = common.get_clock()
    lts = time.clock()
    for step in range(1, EPOCHS + 1):
        xs, ys = next_batch(BATCH_SIZE)

        feed_dict = {x: xs, y_: ys, keep_prob: 0.5}
        _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)

        if step % 20 == 0:
            xs, ys = next_batch(BATCH_SIZE)

            feed_dict = {x: xs, y_: ys, keep_prob: 0.75}
            valid_accuracy = accuracy.eval(feed_dict=feed_dict)
            nts = time.clock()
            print('step {} valid accuracy {}, loss {}({}sec)'.format(step, valid_accuracy, train_loss, round(nts - lts)))
            lts = nts

    end_time = common.get_clock()
    common.get_format_time(start_time, end_time)

    # xs, ys = next_batch(BATCH_SIZE)
    feed_dict = {x: test_images, y_: test_labels, keep_prob: 1.0}
    test_accuracy = accuracy.eval(feed_dict=feed_dict)

    print('test accuracy {}'.format(test_accuracy))
