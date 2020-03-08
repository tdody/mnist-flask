from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import json

from keras.callbacks import ReduceLROnPlateau


batch_size = 32
num_classes = 10
epochs = 60

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    
    Conv2D(32, (5,5), activation='relu', input_shape=(28, 28, 1), padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(32, (5,5), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Flatten(),
    
    Dense(256,activation='relu'),
    Dropout(0.5),
    
    Dense(10,activation='softmax')
])

lrate = ReduceLROnPlateau(monitor='val_accuracy',
                      factor=0.4,
                      patience=3,
                      verbose=1,
                      min_lr=0.0001)
callbacks_list = [lrate]

optimizer = keras.optimizers.Adam(lr=1e-2, beta_1=0.9, beta_2=0.999)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks_list)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
model.save_weights('weights2.h5')