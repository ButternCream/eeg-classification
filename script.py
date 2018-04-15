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
train_one = np.array(a_train)
train_two = np.array(c_train)
train = np.concatenate((train_one, train_two))


labels_tr_one = np.ones((train_one.shape[0], 1)) # create array of ones for those in alc
labels_tr_two = np.zeros((train_two.shape[0], 1)) # create array of 0's for those in control
labels_train = np.concatenate((labels_tr_one, labels_tr_two)) # concat in the same order as x
train = np.expand_dims(x, axis=3)
print("train shape: ", train.shape)
print("labels_train shape: ", labels_train.shape)

# normalize x
train = train / 255

# build our model. The example provided by Boaz/Mathew requires a base model, this builds a model without a base provided. 
model = Sequential()
model.add(Dense(32, input_shape=train.shape[1:]))
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

from keras.callbacks import EarlyStopping
my_callbacks = [EarlyStopping(monitor='acc', patience=5,mode=max)]

model.fit(train, labels_train, epochs=30, batch_size=32, callbacks=my_callbacks)

# simpler format for building your arrays
test_one = np.array(a_test)
test_two = np.array(c_test)
test = np.concatenate((test_one, test_two))


labels_t_one = np.ones((test_one.shape[0], 1)) # create array of ones for those in alc
labels_t_two = np.zeros((test_two.shape[0], 1)) # create array of 0's for those in control
labels_test = np.concatenate((labels_t_one, labels_t_two)) # concat in the same order as x
test = np.expand_dims(x, axis=3)
print("test shape: ", test.shape)
print("labels_test shape: ", labels_test.shape)

# normalize test
test = test / 255


print("Testing using predictions...")
model.predict(test, labels_test, batch_size=32, verbose=1)

