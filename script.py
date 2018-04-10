import os
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.utils import np_utils
from Data import fetch_and_compress, load
from keras.utils import plot_model
# Compress each persons file data into one numpy file


print("Loading")
a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')


# simpler format for building your arrays
x_one = np.array(a_train)
x_two = np.array(c_train)
x = np.concatenate((x_one, x_two))

y_one = np.ones((x_one.shape[0], 1)) # create array of ones for those in alc
y_two = np.zeros((x_two.shape[0], 1)) # create array of 0's for those in control
y = np.concatenate((y_one, y_two)) # concat in the same order as x
x = np.expand_dims(x, axis=3)
print("x shape: ", x.shape)
print("y shape: ", y.shape)

# normalize x
x = x/255

# build our model. The example provided by Boaz/Mathew requires a base model, this builds a model without a base provided. 
model = Sequential()
model.add(Dense(32, input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(Flatten()) # must flatten to ensure we have a 1d array going into our prediction layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("compiling model")
optimizer = keras.optimizers.Adam(lr=0.001) # see keras optimizers for other examples
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
print("model compiled")


model.fit(x, y, epochs=30, batch_size=32)

