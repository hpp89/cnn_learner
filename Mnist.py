import os
import cv2

import numpy as np
import pandas as pd

import keras
from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from keras.initializers import he_normal
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping

weight_delay = 0.001
image_size = 28
num_classes = 10
batch_size = 64
epochs = 50
is_train = False
train_path = './Mnist/train.csv'
test_path = './Mnist/test.csv'
check_point_path = './Mnist/mnist.h5'

train_data = pd.read_csv(train_path)
X_train = train_data.iloc[:, 1:].values
Y_train = train_data.iloc[:, 0].values
Y_train = keras.utils.to_categorical(Y_train, num_classes)
num_train = X_train.shape[0]

X_test = pd.read_csv(test_path).values
num_test = X_test.shape[0]

X_train = X_train.reshape([-1, image_size, image_size, 1])
X_train = X_train / 255.0

X_test = X_test.reshape([-1, image_size, image_size, 1])
X_test = X_test / 255.0

# np.random.seed(42)
# start = -batch_size
#
#
# def next_batch():
#     global X_train
#     global Y_train
#     global start
#
#     start += batch_size
#     if start + batch_size > num_train:
#         order = np.random.permutation(range(num_train))
#         X_train = X_train[order, :, :, :]
#         Y_train = Y_train[order, :]
#
#         start = 0
#
#     return X_train[start:start + batch_size, :, :, :], Y_train[start:start + batch_size, :]

if is_train:
    inputs = keras.Input([image_size, image_size, 1])

    c1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(weight_delay))(inputs)
    p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(weight_delay))(p1)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

    flat = Flatten()(p2)

    d1 = Dense(1024, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(weight_delay))(flat)
    dropout = Dropout(0.5)(d1)

    outputs = Dense(num_classes, activation='softmax', kernel_initializer=he_normal(), kernel_regularizer=l2(weight_delay))(dropout)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    earlystopper = EarlyStopping(patience=0.5, verbose=1)
    checkpoint = ModelCheckpoint(check_point_path, verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True, callbacks=[earlystopper, checkpoint])
else:
    model = load_model(check_point_path)
    Y_test_pred = model.predict(X_test, verbose=1)

    Y_test_pred = np.argmax(Y_test_pred, axis=1)
    sub = pd.DataFrame()
    sub['ImageId'] = [i for i in range(1, num_test + 1)]
    sub['Label'] = Y_test_pred
    sub.to_csv('./Mnist/submission.csv', index=False)
    # n = np.random.randint(0, X_test.shape[0])
    # cv2.imshow('img', X_test[n, :] * 255)
    # print(Y_test_pred[n])
    # cv2.waitKey(-1)

