import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2_layer(x, filter_size, out_num, stride_size, padding='SAME'):
    in_num = x.shape[2].value

    weight = weight_variable([filter_size, filter_size, in_num, out_num])
    bias = bias_variable([out_num])

    conv = tf.nn.conv2d(x, weight, strides=[1, stride_size, stride_size, 1], padding=padding)
    return tf.nn.relu(tf.nn.bias_add(conv, bias))


def max_pool_layer(x, k_size, stride_size, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)


def fully_connection_layer(x, out_num, relu=False, keep_prob=1.0):
    in_num = x.shape[1].value

    weight = weight_variable([in_num, out_num])
    bias = bias_variable([out_num])

    fc = tf.add(tf.matmul(x, weight), bias)

    if relu:
        fc = tf.nn.relu(fc)

    if keep_prob != 1.0:
        fc = tf.nn.dropout(fc, keep_prob=keep_prob)

    return fc