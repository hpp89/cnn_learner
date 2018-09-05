import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import he_normal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt

weight_decay = 0.0001
dropout = 0.5
epochs = 50
iterations = 391
batch_size = 128
num_classes = 10
log_filepath = r'./vgg19_retrain_logs/'


def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 160:
        return 0.01
    return 0.001


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train[:, :, :, 0] = (x_train[:, :, :, 0] - 123.680)
x_train[:, :, :, 1] = (x_train[:, :, :, 1] - 116.779)
x_train[:, :, :, 2] = (x_train[:, :, :, 2] - 103.939)
x_test[:, :, :, 0] = (x_test[:, :, :, 0] - 123.680)
x_test[:, :, :, 1] = (x_test[:, :, :, 1] - 116.779)
x_test[:, :, :, 2] = (x_test[:, :, :, 2] - 103.939)

model = Sequential()

# block1
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), input_shape=[32, 32, 3], name='block1_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block1_pool'))

# block2
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block2_pool'))

# block3
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block3_pool'))

# block4
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), (2, 2), padding='same', name='block4_pool'))

# block5
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv3'))
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


tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

print('Using real-time data augmentation.')
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=iterations,
                              epochs=epochs,
                              callbacks=cbks,
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