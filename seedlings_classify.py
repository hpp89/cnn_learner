import os
import cv2

import numpy as np
import pandas as pd

import keras
from keras.models import Model, load_model, Input
from keras.layers import MaxPooling2D, Conv2D, Dense, Activation
from keras.layers import Flatten, Softmax, BatchNormalization, Dropout
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

CLASS_2_INDEX = {
    'Black-grass': 0,
    'Charlock': 1,
    'Cleavers': 2,
    'Common Chickweed': 3,
    'Common wheat': 4,
    'Fat Hen': 5,
    'Loose Silky-bent': 6,
    'Maize': 7,
    'Scentless Mayweed': 8,
    'Shepherds Purse': 9,
    'Small-flowered Cranesbill': 10,
    'Sugar beet': 11
}

INDEX_2_CLASS = [
    'Black-grass'
    'Charlock',
    'Cleavers',
    'Common Chickweed',
    'Common wheat',
    'Fat Hen',
    'Loose Silky-bent',
    'Maize',
    'Scentless Mayweed',
    'Shepherds Purse',
    'Small-flowered Cranesbill',
    'Sugar beet'
]

train_path = './seedlings/train'
test_path = './seedlings/test'
sub_path = './seedlings/seedlings.csv'

WEIGHT_DELAY = 0.001
IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
NUM_CLASSES = 12
CHECK_POINT_PATH = 'model-seedlings.h5'

X_train = np.ndarray([4750, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
Y_train = []


def get_train_data():
    global Y_train
    global X_train

    count = 0
    for name_dir in os.listdir(train_path):
        sub_name_list = os.listdir('{}/{}'.format(train_path, name_dir))
        Y_train.extend([CLASS_2_INDEX[name_dir]] * len(sub_name_list))
        for sub_name in tqdm(sub_name_list, total=len(sub_name_list)):
            img = cv2.imread('{}/{}/{}'.format(train_path, name_dir, sub_name))
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X_train[count, :, :, :] = img
            count += 1

    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    X_train = X_train / 255.0


X_test = np.ndarray([4750, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
Y_label = []


def get_test_data():
    global X_test
    global Y_label

    count = 0
    for name_dir in os.listdir(test_path):
        Y_label.append(name_dir)
        img = cv2.imread('{}/{}'.format(test_path, name_dir))
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test[count, :, :, :] = img
        count += 1

    X_test = X_test / 255.0


def conv2_block(in_data, filters):
    conv = Conv2D(filters, kernel_size=(3, 3), kernel_initializer=he_normal(), kernel_regularizer=l2(WEIGHT_DELAY), padding='same')(in_data)
    bn = BatchNormalization()(conv)
    out_data = Activation('relu')(bn)

    return out_data


def max_pooling_block(in_data):
    out_data = MaxPooling2D((2, 2), padding='same')(in_data)

    return out_data


def dense_block(in_data, filters, activation='relu'):
    dense = Dense(filters, kernel_initializer=he_normal(), kernel_regularizer=l2(WEIGHT_DELAY))(in_data)
    bn = BatchNormalization()(dense)
    out_data = Activation(activation=activation)(bn)

    return out_data


def get_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
    c1 = conv2_block(inputs, 64)
    c1 = conv2_block(c1, 64)
    p1 = max_pooling_block(c1)

    c2 = conv2_block(p1, 128)
    c2 = conv2_block(c2, 128)
    p2 = max_pooling_block(c2)

    c3 = conv2_block(p2, 256)
    c3 = conv2_block(c3, 256)
    c3 = conv2_block(c3, 256)
    p3 = max_pooling_block(c3)

    flat = Flatten()(p3)
    d1 = dense_block(flat, 128)
    d2 = dense_block(d1, 12, activation='softmax')

    model = Model(inputs=inputs, outputs=d2)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def train():
    get_train_data()

    model = get_model()
    earlystopper = EarlyStopping(patience=0.5, verbose=1)
    checkpointer = ModelCheckpoint(filepath=CHECK_POINT_PATH, verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
              callbacks=[earlystopper, checkpointer])


def test():
    get_test_data()

    model = load_model(CHECK_POINT_PATH)
    Y_test = model.predict(X_test, verbose=1)

    Y_test = np.argmax(Y_test, axis=1)
    output = []
    for i in range(len(Y_test)):
        output.append(INDEX_2_CLASS[Y_test[i]])

    sub = pd.DataFrame({"file": Y_label,
                        "species": output})

    sub.to_csv(sub_path, index=Flatten)


if __name__ == '__main__':
    train()
    # test()
