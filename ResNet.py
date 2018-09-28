import misc
import tensorflow as tf

keep_prob = 0.5


def bottle_neck(inputs, x1_num_out, num_out, scope, training=True, stride=None):
    in_num = inputs.get_shape().as_list()[-1]

    if stride is None:
        stride = 1 if in_num is num_out else 2

    with tf.variable_scope(scope):
        conv_1 = misc.conv2_basic(inputs, kernel=1, stride=stride, out_num=x1_num_out, name='', relu=True)
        norm_1 = misc.batch_normalization(conv_1, training=training)
        conv_2 = misc.conv2_basic(norm_1, kernel=3, out_num=x1_num_out, name='', relu=True)
        norm_2 = misc.batch_normalization(conv_2, training=training)
        conv_3 = misc.conv2_basic(norm_2, kernel=1, out_num=num_out, name='', relu=True)
        norm_3 = misc.batch_normalization(conv_3, training=training)

        if in_num is not num_out:
            conv_4 = misc.conv2_basic(inputs, kernel=1, out_num=num_out, name='', relu=True)
            norm_4 = misc.batch_normalization(conv_4, training=training)
        else:
            norm_4 = inputs

        inputs = misc.leaky_relu(tf.add(norm_3, norm_4))

        return inputs


def block(inputs, num_out, n, init_stride=2, scope="block"):
    with tf.variable_scope(scope):
        x1_num_out = num_out // 4
        inputs = bottle_neck(inputs, x1_num_out, num_out, scope="bottlencek1", stride=init_stride)
        for i in range(1, n):
            inputs = bottle_neck(inputs, x1_num_out, num_out, scope=("bottlencek%s" % (i + 1)))

        return inputs


def ResNet_50(inputs, num_classes=1000, training=True, scope="resnet50"):
    with tf.variable_scope(scope):
        net = misc.conv2_basic(inputs, out_num=64, kernel=7, stride=2)  # -> [batch, 112, 112, 64]
        net = misc.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = misc.max_pooling_2d(net, ksize=3, stride=2)  # -> [batch, 56, 56, 64]
        net = block(net, 256, 3, init_stride=1, scope="block2")  # -> [batch, 56, 56, 256]
        net = block(net, 512, 4, scope="block3")
        net = block(net, 1024, 6, scope="block4")
        net = block(net, 2048, 3, scope="block5")
        net = misc.avg_pooling_2d(net, ksize=7, padding="VALID")  # -> [batch, 1, 1, 2048]
        net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")  # -> [batch, 2048]
        net = misc.fully_connection(net, num_classes)  # -> [batch, num_classes]

        return net
