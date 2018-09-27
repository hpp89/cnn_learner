import misc
import numpy as np
import tensorflow as tf


def VGG_16(inputs, keep_prob, classes_num):
    # block1
    inputs = misc.conv2_basic(inputs, 3, 64, name="conv1_1", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 64, name="conv1_2", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.max_pooling_2x2(inputs, name='pool1_1')

    # block2
    inputs = misc.conv2_basic(inputs, 3, 128, name="conv2_1", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 128, name="conv2_2", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.max_pooling_2x2(inputs, name='pool2_1')

    # block3
    inputs = misc.conv2_basic(inputs, 3, 256, name="conv3_1", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 256, name="conv3_2", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 256, name="conv3_3", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.max_pooling_2x2(inputs, name='pool3_1')

    # block4
    inputs = misc.conv2_basic(inputs, 3, 512, name="conv4_1", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 512, name="conv4_2", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 512, name="conv4_3", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.max_pooling_2x2(inputs, name='pool4_1')

    # block5
    inputs = misc.conv2_basic(inputs, 3, 512, name="conv5_1", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 512, name="conv5_2", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.conv2_basic(inputs, 3, 512, name="conv5_3", padding="SAME", relu=False)
    # inputs = misc.batch_normalization()
    inputs = misc.leak_relu(inputs)

    inputs = misc.max_pooling_2x2(inputs, name='pool5_1')

    shape = inputs.get_shape().as_list()

    inputs = tf.reshape(inputs, shape=[-1, shape[1] * shape[2] * shape[3]])

    inputs = misc.fully_connection(inputs, 4096, keep_prob=keep_prob)
    inputs = misc.fully_connection(inputs, 4096, keep_prob=keep_prob)
    inputs = misc.fully_connection(inputs, classes_num, relu=False)

    return inputs


learning_rate = 0.0005
classes_num = 10
image_size = 32
channels = 3
batch_size = 16
iterables = 1000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = tf.keras.utils.to_categorical(y_train, num_classes=classes_num)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=classes_num)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = misc.images_norm(x_train)
x_test = misc.images_norm(x_test)
x_train, y_train = misc.shuffle(x_train, y_train)

offset = -batch_size


def next_batch():
    global y_train
    global x_train
    global batch_size
    global offset

    offset += batch_size

    num = y_train.shape[0]
    if offset + batch_size > num:
        x_train, y_train = misc.shuffle(x_train, y_train)
        offset = 0

    return x_train[offset:offset + batch_size, :, :, :], y_train[offset:offset + batch_size, :]


x = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, channels])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, classes_num])
keep_prob = tf.placeholder(dtype=tf.float32)

logits = VGG_16(x, keep_prob, classes_num)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

predict = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    now_ts = misc.now_time()
    for i in range(1, iterables + 1):

        xs, ys = next_batch()

        feed_dict = {x: xs, y_: ys, keep_prob: 0.5}
        _, loss_train = sess.run([train_step, loss], feed_dict=feed_dict)

        if i % 20 == 0:
            xs, ys = next_batch()

            feed_dict = {x: xs, y_: ys, keep_prob: 0.75}
            valid_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("step {}, loss_train {}, valid_accuracy {}({}sec)".format(i, round(loss_train), valid_accuracy,
                                                                            misc.diff_time(now_ts)))
            now_ts = misc.now_time()

    feed_dict = {x: x_test[:300, :, :, :], y_: y_test[:300, :], keep_prob: 1.0}
    test_accuracy = accuracy.eval(feed_dict=feed_dict)
    print("test_accuracy {}".format(test_accuracy))
