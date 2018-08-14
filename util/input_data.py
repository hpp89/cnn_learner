import util
import numpy as np
import pandas as pd


def get_cifar_data():
    train_names = []
    train_images = []
    train_labels = []
    for i in range(5):
        name = 'data_batch_%g' % (i + 1)
        d = util.unpickle('./../dataSets/cifar-10-python/cifar-10-batches-py/%s' % name)
        train_names = train_names + d[b'filenames']
        if i == 0:
            train_images = d[b'data']
        else:
            train_images = np.concatenate((train_images, d[b'data']))
        train_labels = train_labels + d[b'labels']

    d = util.unpickle('./../dataSets/cifar-10-python/cifar-10-batches-py/test_batch')
    test_names = d[b'filenames']
    test_images = d[b'data']
    test_labels = d[b'labels']

    return train_images, train_labels, test_images, test_labels


def get_ImageNet():
    file_path = './../dataSets/imagenet_fall11_urls/fall11_urls.txt'

    with open(file_path, 'r') as f:
        d = [line for line in f]

get_ImageNet()


