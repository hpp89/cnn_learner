import misc
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
    inputs = misc.fully_connection(inputs, classes_num)

    return inputs
