import numpy as np
import tensorflow as tf
import misc


def inception_v1(inputs, out_size_1, out_size_3_reduce, out_size_3, out_size_5_reduce, out_size_5,
                 out_size_max_pool_proj):
    conv_1_1 = misc.conv2_basic(inputs, 1, out_size_1, padding='VALID')

    conv_3_1 = misc.conv2_basic(inputs, 1, out_size_3_reduce, padding='VALID')
    conv_3_2 = misc.conv2_basic(conv_3_1, 3, out_size_3, padding='VALID')

    conv_5_1 = misc.conv2_basic(inputs, 1, out_size_5_reduce, padding='VALID')
    conv_5_2 = misc.conv2_basic(conv_5_1, 5, out_size_5,  padding='VALID')

    conv_max_pool_1 = misc.max_pooling_2d(inputs, 3, 1, padding='VALID')
    conv_max_pool_2 = misc.conv2_basic(conv_max_pool_1, 1, out_size_max_pool_proj, padding='VALID')

    inputs = tf.concat((conv_1_1, conv_3_2, conv_5_2, conv_max_pool_2), axis=3)

    return inputs


def inception_v2(inputs, out_size_1, out_size_3_reduce, out_size_3, out_size_5_reduce, out_size_5_1, out_size_5_2,
                 out_size_max_pool_proj):
    conv_1_1 = misc.conv2_basic(inputs, 1, out_size_1, padding='VALID')

    conv_3_1 = misc.conv2_basic(inputs, 1, out_size_3_reduce, padding='VALID')
    conv_3_2 = misc.conv2_basic(conv_3_1, 3, out_size_3, padding='VALID')

    conv_5_1 = misc.conv2_basic(inputs, 1, out_size_5_reduce, padding='VALID')
    conv_5_2 = misc.conv2_basic(conv_5_1, 3, out_size_5_1, padding='VALID')
    conv_5_3 = misc.conv2_basic(conv_5_2, 3, out_size_5_2, padding='VALID')

    conv_max_pool_1 = misc.max_pooling_2d(inputs, 3, 1, padding='VALID')
    conv_max_pool_2 = misc.conv2_basic(conv_max_pool_1, 1, out_size_max_pool_proj, padding='VALID')

    inputs = tf.concat((conv_1_1, conv_3_2, conv_5_3, conv_max_pool_2), axis=3)

    return inputs


def inception_v3():
    pass

