import os
import time
import tensorflow as tf


def get_clock():
    print('现在时间: %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    return time.clock()


def get_format_time(start_time, over_time):
    ts = int(over_time - start_time)
    days = int(ts / 3600 / 24)
    hours = int(ts / 3600) - days * 24
    minutes = int((ts % 3600) / 60)
    seconds = ts % 60

    print('%g天%g小时%g分%g秒' % (days, hours, minutes, seconds))


def get_is_training(save_path, trainable=False):
    if trainable:
        return True

    return not os.path.exists(save_path + '.meta')


def saver():
    return tf.train.Saver()
