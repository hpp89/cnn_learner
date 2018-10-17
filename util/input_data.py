import os
import cv2
import misc
import util as ul
import numpy as np
import pandas as pd


def get_cifar_data():
    train_names = []
    train_images = []
    train_labels = []
    for i in range(5):
        name = 'data_batch_%g' % (i + 1)
        d = ul.unpickle('./../dataSets/cifar-10-python/cifar-10-batches-py/%s' % name)
        train_names = train_names + d[b'filenames']
        if i == 0:
            train_images = d[b'data']
        else:
            train_images = np.concatenate((train_images, d[b'data']))
        train_labels = train_labels + d[b'labels']

    d = ul.unpickle('./../dataSets/cifar-10-python/cifar-10-batches-py/test_batch')
    test_names = d[b'filenames']
    test_images = d[b'data']
    test_labels = d[b'labels']

    return train_images, train_labels, test_images, test_labels


def dog_breed_train(image_size):
    images = os.listdir('./../dataSets/DogBreed/train')
    labels = pd.read_csv('./../dataSets/DogBreed/labels.csv')

    num = len(images)

    data = np.ndarray(shape=[num, image_size, image_size, 3])
    for i in range(num):
        img = cv2.imread('./../dataSets/DogBreed/train/{}'.format(images[i]))
        img = cv2.resize(img, dsize=(image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data[i, :, :, :] = img

    data = misc.images_norm(data)
    labels = pd.get_dummies(labels.iloc[:, 1].values)
    return data, labels


def dog_breed_test():
    images = os.listdir('./../dataSets/DogBreed/test')

    return images
