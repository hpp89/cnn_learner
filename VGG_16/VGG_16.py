import util.network as net
import tensorflow as tf
import util.common as common
import tensorflow.contrib.slim as slim

CHANNELS = 3
BATCH_SIZE = 64
IMAGE_SIZE = 224
LEARNING_RATE = 0.0001


def next_batch():
    pass


def VGG_16(x, keep_prob):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])

    with tf.name_scope('layer1'):
        x = net.conv2_layer(x, 3, 64, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 64, 1, padding='SAME')
        x = net.max_pool_layer(x, 2, 2, padding='VALID')

    with tf.name_scope('layer2'):
        x = net.conv2_layer(x, 3, 128, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 128, 1, padding='SAME')
        x = net.max_pool_layer(x, 2, 2, padding='VALID')

    with tf.name_scope('layer3'):
        x = net.conv2_layer(x, 3, 256, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 256, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 256, 1, padding='SAME')
        x = net.max_pool_layer(x, 2, 2, padding='VALID')

    with tf.name_scope('layer4'):
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.max_pool_layer(x, 2, 2, padding='VALID')

    with tf.name_scope('layer5'):
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.conv2_layer(x, 3, 512, 1, padding='SAME')
        x = net.max_pool_layer(x, 2, 2, padding='VALID')

    with tf.name_scope('flat'):
        shape = x.shape
        x = tf.reshape(x, [-1, shape[1].value * shape[2].value * shape[3].value])

    with tf.name_scope('layer6'):
        x = net.fully_connection_layer(x, 4096, relu=True, keep_prob=keep_prob)
        x = net.fully_connection_layer(x, 4096, relu=True, keep_prob=keep_prob)
        x = net.fully_connection_layer(x, 1000, relu=False, keep_prob=1.0)

    return x


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE * CHANNELS])
y_ = tf.placeholder(tf.float32, shape=[None, 1000])

hat = VGG_16(x, keep_prob)

y = tf.nn.softmax(hat)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), dtype=tf.float32))

EPOCHS = 1000

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
