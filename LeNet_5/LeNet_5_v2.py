import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Flatten
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import math
import matplotlib.pyplot as plt
import misc
import numpy as np

weight_decay = 0.0001
num_classes = 10
dropout = 0.5
image_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = misc.images_norm(x_train.astype(np.float32))
x_test = misc.images_norm(x_test.astype(np.float32))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def conv(filters, kernel_size):
    return Conv2D(filters, input_shape=[32, 32, 3], kernel_size=kernel_size, strides=(1, 1), padding='same', kernel_initializer=he_normal(),
                  kernel_regularizer=keras.regularizers.l2(weight_decay))


model = Sequential()

model.add(conv(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))

model.add(conv(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(strides=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(1024, use_bias=True, kernel_initializer=he_normal(), kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1024, use_bias=True, kernel_initializer=he_normal(), kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_classes, use_bias=True, kernel_initializer=he_normal(), kernel_regularizer=keras.regularizers.l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])


def scheduler(epoch):
    if epoch < 2:
        return 0.1
    if epoch < 3:
        return 0.01
    if epoch < 4:
        return 0.001
    return 0.0001


change_lr = LearningRateScheduler(schedule=scheduler)
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

datagen.fit(x_train)

epochs = 40
batch_size = 64
iterations = math.floor(x_train.shape[0] / batch_size)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=iterations,
                              epochs=epochs,
                              callbacks=[change_lr],
                              validation_data=(x_test, y_test))

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
