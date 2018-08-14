import util.common as common
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

EPOCHS = 5000
BATCH_SIZE = 64
TEST_SIZE = 2000
LEARNING_RATE = 0.0001

mnist = input_data.read_data_sets('./../dataSets/Mnist', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def LeNet_5(x):
    with tf.name_scope('reshape'):
        x = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([3, 3, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x, w_conv1), b_conv1))

    with tf.name_scope('pool1'):
        pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([3, 3, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(pool1, w_conv2), b_conv2))

    with tf.name_scope('pool2'):
        pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('flat'):
        shape = pool2.shape
        pool2_flat = tf.reshape(pool2, [-1, shape[1].value * shape[2].value * shape[3].value])

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([pool2_flat.shape[1].value, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.add(tf.matmul(pool2_flat, w_fc1), b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(dtype=tf.float32)
        dropout = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        h_fc2 = tf.add(tf.matmul(dropout, w_fc2), b_fc2)

    return h_fc2, keep_prob


x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])

h_conv, keep_prob = LeNet_5(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h_conv, labels=y_))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

predict = tf.equal(tf.argmax(h_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, dtype=tf.float32))

save_path = './LeNet_5.ckpt'
is_training = common.get_is_training(save_path)
saver = common.saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if is_training:
        start_time = common.get_clock()

        for step in range(1, EPOCHS + 1):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            feed_dict = {x: xs, y_: ys, keep_prob: 0.5}
            _, cost = sess.run([train_step, loss], feed_dict=feed_dict)

            if step % 10 == 0:
                xs, ys = mnist.validation.next_batch(BATCH_SIZE)

                feed_dict = {x: xs, y_: ys, keep_prob: 0.75}
                validation_accuracy = accuracy.eval(feed_dict=feed_dict)

                print('step {}, loss {}, validation_accuracy {}'.format(step, cost, validation_accuracy))

        saver.save(sess, save_path)

        end_time = common.get_clock()
        common.get_format_time(start_time, end_time)
    else:

        saver.restore(sess, save_path)

    xs, ys = mnist.test.next_batch(TEST_SIZE)

    feed_dict = {x: xs, y_: ys, keep_prob: 1.0}
    test_accuracy = accuracy.eval(feed_dict=feed_dict)

    print('test_accuracy {}'.format(test_accuracy))
