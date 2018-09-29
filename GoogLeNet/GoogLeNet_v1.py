import misc
import tensorflow as tf


def inception_v1(inputs, out_size_1, out_size_3_reduce, out_size_3, out_size_5_reduce, out_size_5,
                 out_size_max_pool_proj):
    conv_1_1 = misc.conv2_basic(inputs, 1, out_size_1, padding='VALID')

    conv_3_1 = misc.conv2_basic(inputs, 1, out_size_3_reduce, padding='VALID')
    conv_3_2 = misc.conv2_basic(conv_3_1, 3, out_size_3, padding='VALID')

    conv_5_1 = misc.conv2_basic(inputs, 1, out_size_5_reduce, padding='VALID')
    conv_5_2 = misc.conv2_basic(conv_5_1, 5, out_size_5, padding='VALID')

    conv_max_pool_1 = misc.max_pooling_2d(inputs, 3, 1, padding='VALID')
    conv_max_pool_2 = misc.conv2_basic(conv_max_pool_1, 1, out_size_max_pool_proj, padding='VALID')

    inputs = tf.concat((conv_1_1, conv_3_2, conv_5_2, conv_max_pool_2), axis=3)

    return inputs


def GoogLeNet_v1(inputs, keep_prob, num_classes):
    conv_1 = misc.conv2_basic(inputs, kernel=7, out_num=64, stride=2)
    max_pool_1 = misc.max_pooling_2d(conv_1, ksize=3, stride=2)
    conv_2 = misc.conv2_basic(max_pool_1, kernel=1, out_num=192, stride=1)
    conv_2_1 = misc.conv2_basic(conv_2, kernel=3, out_num=192, stride=1)
    max_pool_2 = misc.max_pooling_2d(conv_2_1, ksize=3, stride=2)

    # inception(3)
    inception_3a = inception_v1(max_pool_2, out_size_1=64, out_size_3_reduce=96, out_size_3=128, out_size_5_reduce=16, out_size_5=32,
                                out_size_max_pool_proj=32)

    inception_3b = inception_v1(inception_3a, out_size_1=128, out_size_3_reduce=128, out_size_3=192, out_size_5_reduce=32, out_size_5=96,
                                out_size_max_pool_proj=64)

    max_pool_3 = misc.max_pooling_2d(inception_3b, ksize=3, stride=2)

    # inception(4)
    inception_4a = inception_v1(max_pool_3, out_size_1=192, out_size_3_reduce=96, out_size_3=208, out_size_5_reduce=16, out_size_5=48,
                                out_size_max_pool_proj=64)

    inception_4b = inception_v1(inception_4a, out_size_1=160, out_size_3_reduce=112, out_size_3=224, out_size_5_reduce=24, out_size_5=64,
                                out_size_max_pool_proj=64)

    inception_4c = inception_v1(inception_4b, out_size_1=128, out_size_3_reduce=128, out_size_3=256, out_size_5_reduce=24, out_size_5=64,
                                out_size_max_pool_proj=64)

    inception_4d = inception_v1(inception_4c, out_size_1=112, out_size_3_reduce=144, out_size_3=288, out_size_5_reduce=32, out_size_5=64,
                                out_size_max_pool_proj=64)

    inception_4e = inception_v1(inception_4d, out_size_1=256, out_size_3_reduce=160, out_size_3=320, out_size_5_reduce=32, out_size_5=128,
                                out_size_max_pool_proj=128)

    max_pool_4 = misc.max_pooling_2d(inception_4e, ksize=3, stride=2)

    # inception(5)
    inception_5a = inception_v1(max_pool_4, out_size_1=256, out_size_3_reduce=160, out_size_3=320, out_size_5_reduce=32, out_size_5=128,
                                out_size_max_pool_proj=128)

    inception_5b = inception_v1(inception_5a, out_size_1=384, out_size_3_reduce=192, out_size_3=384, out_size_5_reduce=48, out_size_5=128,
                                out_size_max_pool_proj=128)

    avg_pool = misc.avg_pooling_2d(inception_5b, ksize=7, stride=1)
    dropout = tf.nn.dropout(avg_pool, keep_prob=keep_prob)
    fc = misc.fully_connection(dropout, out_num=num_classes)
    output = tf.nn.softmax(fc)

    return output
