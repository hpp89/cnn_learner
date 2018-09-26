import numpy as np
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(dtype=tf.float32, stddev=0.1, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


def leak_relu(inputs, alpha=0.0):
    return tf.maximum(inputs * alpha, inputs)


def batch_normalization(inputs, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration)
    bn_epsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(inputs, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(inputs, [0])

    update_moving_averages = exp_moving_avg.apply([mean, variance])
    mean = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    variance = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    inputs = tf.nn.batch_normalization(inputs, mean, variance, offset, None, bn_epsilon)

    return inputs, update_moving_averages


def conv2_basic(inputs, kernel, out_num, name, stride=1, padding="SAME", relu=True, alpha=0.0):
    in_num = inputs.get_shape().as_list()[3]

    weight = weight_variable([kernel, kernel, in_num, out_num])
    bias = bias_variable([out_num])

    inputs = tf.nn.conv2d(inputs, filter=weight, strides=[1, stride, stride, 1], name=name, padding=padding)
    inputs = tf.nn.bias_add(inputs, bias)

    if relu:
        inputs = leak_relu(inputs, alpha=alpha)

    return inputs


def max_pooling_2x2(inputs, name, ksize=2, stride=2, padding="SAME"):
    return tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding, name=name)


def fully_connection(inputs, out_num, keep_prob=1.0, relu=True, alpha=0.0):
    in_num = inputs.get_shape().as_list()[-1]

    weight = weight_variable(shape=[in_num, out_num])
    bias = bias_variable([out_num])

    inputs = tf.matmul(inputs, weight) + bias

    if relu:
        inputs = leak_relu(inputs, alpha=alpha)

    if keep_prob < 1.0:
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    return inputs
