import numpy as np
import tensorflow as tf
import util.network as net


def inception_v1(inputs, out_size_1, out_size_3_reduce, out_size_3, out_size_5_reduce, out_size_5,
                 out_size_max_pool_proj):
    conv_1_1 = net.conv2_layer(inputs, 1, out_size_1, 1, padding='VALID')

    conv_3_1 = net.conv2_layer(inputs, 1, out_size_3_reduce, 1, padding='VALID')
    conv_3_2 = net.conv2_layer(conv_3_1, 3, out_size_3, 1, padding='VALID')

    conv_5_1 = net.conv2_layer(inputs, 1, out_size_5_reduce, 1, padding='VALID')
    conv_5_2 = net.conv2_layer(conv_5_1, 5, out_size_5, 1, padding='VALID')

    conv_max_pool_1 = net.max_pool_layer(inputs, 3, 1, padding='VALID')
    conv_max_pool_2 = net.conv2_layer(conv_max_pool_1, 1, out_size_max_pool_proj, 1, padding='VALID')

    inputs = tf.concat((conv_1_1, conv_3_2, conv_5_2, conv_max_pool_2), axis=3)

    return inputs


def inception_v2(inputs, out_size_1, out_size_3_reduce, out_size_3, out_size_5_reduce, out_size_5_1, out_size_5_2,
                 out_size_max_pool_proj):
    conv_1_1 = net.conv2_layer(inputs, 1, out_size_1, 1, padding='VALID')

    conv_3_1 = net.conv2_layer(inputs, 1, out_size_3_reduce, 1, padding='VALID')
    conv_3_2 = net.conv2_layer(conv_3_1, 3, out_size_3, 1, padding='VALID')

    conv_5_1 = net.conv2_layer(inputs, 1, out_size_5_reduce, 1, padding='VALID')
    conv_5_2 = net.conv2_layer(conv_5_1, 3, out_size_5_1, 1, padding='VALID')
    conv_5_3 = net.conv2_layer(conv_5_2, 3, out_size_5_2, 1, padding='VALID')

    conv_max_pool_1 = net.max_pool_layer(inputs, 3, 1, padding='VALID')
    conv_max_pool_2 = net.conv2_layer(conv_max_pool_1, 1, out_size_max_pool_proj, 1, padding='VALID')

    inputs = tf.concat((conv_1_1, conv_3_2, conv_5_3, conv_max_pool_2), axis=3)

    return inputs


def inception_v3():
    pass

