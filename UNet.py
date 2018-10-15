import misc
import tensorflow as tf


def UNet(inputs):
    result = dict()

    with tf.name_scope('block_1'):
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=64, stride=1, padding='SAME', name='block_1_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=64, stride=1, padding='SAME', name='block_1_conv2')
        result['block_1'] = inputs
        inputs = misc.max_pooling_2d(inputs, ksize=2, stride=2, padding='SAME', name='block_1_maxpooling')

    with tf.name_scope('block_2'):
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=128, stride=1, padding='SAME', name='block_2_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=128, stride=1, padding='SAME', name='block_2_conv2')
        result['block_2'] = inputs
        inputs = misc.max_pooling_2d(inputs, ksize=2, stride=2, padding='SAME', name='block_2_maxpooling')

    with tf.name_scope('block_3'):
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=256, stride=1, padding='SAME', name='block_3_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=256, stride=1, padding='SAME', name='block_3_conv2')
        result['block_3'] = inputs
        inputs = misc.max_pooling_2d(inputs, ksize=2, stride=2, padding='SAME', name='block_3_maxpooling')

    with tf.name_scope('block_4'):
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=512, stride=1, padding='SAME', name='block_4_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=512, stride=1, padding='SAME', name='block_4_conv2')
        result['block_4'] = inputs
        inputs = misc.max_pooling_2d(inputs, ksize=2, stride=2, padding='SAME', name='block_4_maxpooling')

    with tf.name_scope('block_5'):
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=1024, stride=1, padding='SAME', name='block_5_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=1024, stride=1, padding='SAME', name='block_5_conv2')

    with tf.name_scope('block_6'):
        shape = inputs.get_shape().as_list()
        shape[1] = 56
        shape[2] = 56
        shape[3] = 512
        inputs = misc.conv2_transpose(inputs, kernel=2, out_shape=shape, stride=1, padding='SAME', name='block_6_conv_transpose')
        ret4 = misc.image_crop(result['block_4'], [56, 56])

        inputs = tf.concat(values=[ret4, inputs], axis=3)

        inputs = misc.conv2_basic(inputs, kernel=3, out_num=512, stride=1, padding='SAME', name='block_6_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=512, stride=1, padding='SAME', name='block_6_conv2')

    with tf.name_scope('block_7'):
        shape = inputs.get_shape().as_list()
        shape[1] = 104
        shape[2] = 104
        shape[3] = 256
        inputs = misc.conv2_transpose(inputs, kernel=2, out_shape=shape, stride=1, padding='SAME', name='block_7_conv_transpose')
        ret3 = misc.image_crop(result['block_3'], [104, 104])

        inputs = tf.concat(values=[ret3, inputs], axis=3)

        inputs = misc.conv2_basic(inputs, kernel=3, out_num=256, stride=1, padding='SAME', name='block_7_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=256, stride=1, padding='SAME', name='block_7_conv2')

    with tf.name_scope('block_8'):
        shape = inputs.get_shape().as_list()
        shape[1] = 200
        shape[2] = 200
        shape[3] = 128
        inputs = misc.conv2_transpose(inputs, kernel=2, out_shape=shape, stride=1, padding='SAME', name='block_8_conv_transpose')
        ret2 = misc.image_crop(result['block_2'], [200, 200])

        inputs = tf.concat(values=[ret2, inputs], axis=3)

        inputs = misc.conv2_basic(inputs, kernel=3, out_num=128, stride=1, padding='SAME', name='block_8_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=128, stride=1, padding='SAME', name='block_8_conv2')

    with tf.name_scope('block_9'):
        shape = inputs.get_shape().as_list()
        shape[1] = 392
        shape[2] = 392
        shape[3] = 64
        inputs = misc.conv2_transpose(inputs, kernel=2, out_shape=shape, stride=1, padding='SAME', name='block_9_conv_transpose')
        ret1 = misc.image_crop(result['block_1'], [392, 392])

        inputs = tf.concat(values=[ret1, inputs], axis=3)

        inputs = misc.conv2_basic(inputs, kernel=3, out_num=64, stride=1, padding='SAME', name='block_9_conv1')
        inputs = misc.conv2_basic(inputs, kernel=3, out_num=64, stride=1, padding='SAME', name='block_9_conv2')
        inputs = misc.conv2_basic(inputs, kernel=1, out_num=1, stride=1, padding='SAME', name='block_9_conv3', relu=False)

        inputs = tf.nn.softmax(inputs)
        return inputs
