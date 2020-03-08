import tensorflow as tf
import numpy as np
import keras.models

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from scipy.misc import imread, imresize,imshow

def load_model():

    # parameters
    n_classes = 10
    image_width, image_height = 28, 28
    input_shape = (img_rows, img_cols, 1)

    # create empty model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # load weights
    model.load_weights('weights.h5')

    # compile model
    model.compile(oss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    graph = tf.get_default_graph()

    return model, graph