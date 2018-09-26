import keras
import time
import numpy as np
import tensorflow as tf
from keras.datasets import
from keras.layers import MaxPooling2D, Dense, Activation, Flatten, Conv2D
from keras.initializers import he_normal
from keras.layers import BatchNormalization, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

epochs = 15
iterations = 391
batch_size = 128
num_classes = 10
dropout = 0.5
weight_decay = 0.0001

log_file_path = './alex_net_logs'


def scheduler(epoch):
    if epoch < 10:
        return 0.1
    if epoch < 13:
        return 0.01
    return 0.001


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
train_images = train_images.astype(np.float32)
test_images = test_images.astype(np.float32)

train_images[:, :, :, 0] = (train_images[:, :, :, 0] - 123.680)
train_images[:, :, :, 1] = (train_images[:, :, :, 1] - 116.779)
train_images[:, :, :, 2] = (train_images[:, :, :, 2] - 103.939)
test_images[:, :, :, 0] = (test_images[:, :, :, 0] - 123.680)
test_images[:, :, :, 1] = (test_images[:, :, :, 1] - 116.779)
test_images[:, :, :, 2] = (test_images[:, :, :, 2] - 103.939)

print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)

model = keras.models.Sequential()
# block1
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv', input_shape=[32, 32, 3]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block1_pool'))

# block2
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block2_pool'))

# block3
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block3_pool'))

# block4
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block4_pool'))

# block5
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block5_pool'))

model.add(Flatten(name='faltten'))
model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(4096, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(10, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay), kernel_initializer=he_normal(), name='fc_3'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

tb_cb = TensorBoard(log_dir=log_file_path, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

datagen.fit(train_images)

history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                              steps_per_epoch=iterations,
                              epochs=epochs,
                              callbacks=cbks,
                              validation_data=(test_images, test_labels))
model.save('retrain.h5')

loss = history.history.get('loss')
acc = history.history.get('acc')
val_loss = history.history.get('val_loss')
val_acc = history.history.get('val_acc')

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss, label="Training")
plt.plot(range(len(val_loss)), val_loss, label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc, label='Training')
plt.plot(range(len(val_acc)), val_acc, label='Validation')
plt.title('Accuracy')
plt.show()
