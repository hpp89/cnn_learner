import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Lambda, Conv2DTranspose, Input
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import he_normal
from keras import backend as K

import tensorflow as tf

seed = 42
random.seed = seed
np.random.seed(seed)
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

IMAGE_SIZE = 128
CHANNELS = 3
DROPOUT = [0.1, 0.2, 0.3]

TRAIN_PATH = './stage1_train/'
TEST_PATH = './stage2_test_final/'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

print(train_ids)

X_train = np.zeros((len(train_ids), IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.bool)

sys.stdout.flush()
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
#     path = TRAIN_PATH + id_
#     img = imread(path + '/images/' + id_ + '.png')[:, :, :CHANNELS]
#     img = resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant', preserve_range=True)
#     X_train[n] = img
#     mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.bool)
#     for mask_file in next(os.walk(path + '/masks/'))[2]:
#         mask_ = imread(path + '/masks/' + mask_file)
#         mask_ = np.expand_dims(resize(mask_, (IMAGE_SIZE, IMAGE_SIZE), mode='constant',
#                                       preserve_range=True), axis=-1)
#         mask = np.maximum(mask, mask_)
#     Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# inputs = Input((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
# s = Lambda(lambda x: x / 255.0)(inputs)
#
# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(s)
# c1 = Dropout(DROPOUT[0])(c1)
# c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c1)
# p1 = MaxPooling2D((2, 2))(c1)
#
# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(p1)
# c2 = Dropout(DROPOUT[0])(c2)
# c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c2)
# p2 = MaxPooling2D((2, 2))(c2)
#
# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(p2)
# c3 = Dropout(DROPOUT[1])(c3)
# c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c3)
# p3 = MaxPooling2D((2, 2))(c3)
#
# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(p3)
# c4 = Dropout(DROPOUT[1])(c4)
# c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c4)
# p4 = MaxPooling2D((2, 2))(c4)
#
# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(p4)
# c5 = Dropout(DROPOUT[2])(c5)
# c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c5)
#
# u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
# u6 = concatenate([u6, c4])
# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(u6)
# c6 = Dropout(DROPOUT[1])(c6)
# c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c6)
#
# u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
# u7 = concatenate([u7, c3])
# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(u7)
# c7 = Dropout(DROPOUT[1])(c7)
# c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c7)
#
# u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
# u8 = concatenate([u8, c2])
# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(u8)
# c8 = Dropout(DROPOUT[0])(c8)
# c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c8)
#
# u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
# u9 = concatenate([u9, c1])
# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(u9)
# c9 = Dropout(DROPOUT[0])(c9)
# c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer=he_normal(), padding='same')(c9)
#
# outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
#
# model = Model(inputs=[inputs], outputs=[outputs])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
# model.summary()
#
# earlystopper = EarlyStopping(patience=0.5, verbose=1)
# checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
# results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
#                     callbacks=[earlystopper, checkpointer])

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
# preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

# ix = random.randint(0, len(preds_train_t))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-2.csv', index=False)
