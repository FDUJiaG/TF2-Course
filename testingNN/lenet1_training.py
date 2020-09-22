#!/usr/bin/env python
# coding: utf-8

# # Calculating Information Entropy after MNIST Dataset Prediction

# ## Loading Packages and Settings

import os
# # GPU Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import time

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
import keras.backend as K


K.image_data_format() == "tf"
seed = 0
np.random.seed(seed)


# ## MNIST Dataset
# ### View Data Dimensions

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("X Train Shape:", x_train.shape, "\nY Train Shape:", y_train.shape)
print("X Test  Shape:", x_test.shape, "\nY Test  Shape:", y_test.shape)

img_rows, img_cols = 28, 28
num_classes = 10

# ### Reshape the Data

# Reshape to be samples * pixels * width * height
x_train = x_train.reshape(
    x_train.shape[0], img_rows, img_cols, 1
).astype('float32')
x_test = x_test.reshape(
    x_test.shape[0], img_rows, img_cols, 1
).astype('float32')
print("X Train ReShape:", x_train.shape)
print("X Test  ReShape:", x_test.shape)
print("DateType:", x_train.dtype)

# ### Data Normalization

print("Max Pixel Value:", np.max(x_train), "\nMin Pixel Value:", np.min(x_train))

# Normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

print("Max Adjust Pixel Value:", np.max(x_train), "\nMin Adjust Pixel Value:", np.min(x_train))

# ### One-Hot Encode

y_train = np_utils.to_categorical(y_train, num_classes)  # Converts a class vector (integers) to binary class matrix.
y_test = np_utils.to_categorical(y_test, num_classes)
print("Y Train ReShape:", y_train.shape)
print("Y Test  ReShape:", y_test.shape)


# ## LeNet1
# ### Define the Model

data_input_shape = (img_rows, img_cols, 1)

# Create Model
model = Sequential(name="LeNet1")
model.add(Conv2D(
    32, (5, 5),
    input_shape=data_input_shape,
    activation='relu',
    padding="same",
    name="Input"
))
model.add(MaxPooling2D(
    pool_size=(2, 2), name="MaxPooling_1"
))
model.add(Conv2D(
    16, (3, 3), 
    activation='relu', 
    padding="same",
    name="Conv2D"
))
model.add(MaxPooling2D(
    pool_size=(2, 2), name="MaxPooling_2"
))
# model.add(Dropout(0.2, name="Dropout"))
model.add(Flatten(name="Flatten"))
# model.add(Dense(
#     128, activation='relu', name="Dense_1"
# ))
# model.add(Dense(
#     64, activation='relu', name="Dense_2"
# ))
model.add(Dense(
    num_classes, 
    activation='softmax', 
    name="Output"
))

# See the Model Summary
model.summary()

# ### Training the Model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 256
epochs = 10
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

# ### Model Evaluation

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')

evaluate = model.evaluate(x_test, y_test, verbose=1)

# ### Save the Model

model_dir = os.path.join(os.path.abspath('.'), "models")
model_name = "lenet1.h5"

if not os.path.exists(model_dir):
    # If models directory does not exist, create a directory
    os.makedirs(model_dir) 
    print("[INFO]", model_dir, "is Successfully Created!")

model.save(os.path.join(model_dir, model_name))
print("[INFO]", model_name, "Model File is Successfully Saved!")

plt.show()
