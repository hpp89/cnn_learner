import time
import math
import numpy as np
import tensorflow as tf

np.random.seed(133)


def diff_time(start_time):
    now = now_time()
    return math.floor((now - start_time) * 10) / 10


def now_time():
    return time.clock()


def images_norm(images, name='mean'):
    if name == 'mean':
        images[:, :, 0] -= 123.680
        images[:, :, 1] -= 116.779
        images[:, :, 2] -= 103.939
    elif name == 'norm_0_1':
        images = images / 255.0
    elif name == 'norm__1_1':
        images = images / 255.0 * 2 - 1.0

    return images


def shuffle(x, y):
    num = x.shape[0]

    order = np.random.permutation(range(num))
    x = x[order, :, :, :]
    y = y[order, :]

    return x, y


# xavier初始化器，把权重初始化在low和high范围内(满足N(0,2/Nin+Nout))
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def weight_variable(shape):
    initial = tf.truncated_normal(dtype=tf.float32, stddev=0.1, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial)


def leaky_relu(inputs, alpha=0.0):
    return tf.nn.leaky_relu(inputs, alpha)


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training, momentum=0.9)


def conv2_basic(inputs, kernel, out_num, name=None, stride=1, padding="SAME", relu=True, alpha=0.0):
    in_num = inputs.get_shape().as_list()[3]

    weight = weight_variable([kernel, kernel, in_num, out_num])
    bias = bias_variable([out_num])

    inputs = tf.nn.conv2d(inputs, filter=weight, strides=[1, stride, stride, 1], name=name, padding=padding)
    inputs = tf.nn.bias_add(inputs, bias)

    if relu:
        inputs = leaky_relu(inputs, alpha=alpha)

    return inputs


def conv2_transpose(inputs, kernel, out_shape, name=None, stride=1, padding='SAME'):
    shape = inputs.get_shape().as_list()

    weight = weight_variable([kernel, kernel, out_shape[3], shape[3]])
    bias = bias_variable([out_shape[3]])

    inputs = tf.nn.conv2d_transpose(inputs, filter=weight, output_shape=out_shape, strides=[1, stride, stride, 1], padding=padding,
                                    name=name)
    inputs = tf.nn.bias_add(inputs, bias)

    return inputs


def max_pooling_2d(inputs, ksize=2, stride=2, padding="SAME", name=None):
    return tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding,
                          name=name)


def avg_pooling_2d(inputs, ksize=2, stride=2, padding="SAME", name=None):
    return tf.nn.avg_pool(inputs, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding,
                          name=name)


def fully_connection(inputs, out_num, keep_prob=1.0, relu=True, alpha=0.0):
    in_num = inputs.get_shape().as_list()[-1]

    weight = xavier_init(in_num, out_num)
    bias = bias_variable([out_num])

    inputs = tf.add(tf.matmul(inputs, weight), bias)

    if relu:
        inputs = leaky_relu(inputs, alpha=alpha)

    if keep_prob is not 1.0:
        inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

    return inputs


def image_crop(inputs, crop_size):
    return tf.image.crop_and_resize(inputs, boxes=[[0, 0, 1, 1]], box_ind=[0], crop_size=crop_size)
