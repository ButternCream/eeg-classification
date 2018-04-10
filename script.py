import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.utils import np_utils
from Data import fetch_and_compress, load
from keras.utils import plot_model
# Compress each persons file data into one numpy file
#print("Compressing data")
#fetch_and_compress(['./alcoholic/', './control/'])
print("Loading")
#load(['./alcohol_compressed/', './control_compressed/'])
a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
#print("Dimensions atrain: ", len(a_train), "x", len(a_train[0]))
print('Loaded')

trainLabels = [1 for i in range(len(a_train))] + [0 for j in range(len(c_train))]
#trainLabels = [1 for i in range(int(1/3*len(c_train)))] + [0 for j in range(int(2/3*len(c_train))+ 1)]
'''
train =  a_train[:]
for c in c_train:
    train.append(c)
'''
train = a_train[:]
for c in c_train:
    train.append(c)

t = np.array(train)
t = np.expand_dims(t, axis=3)
print("shape t = ", t.shape)



# print('Declaring model')
# model = Sequential()
# model.add(Dense(1, activation='relu', input_shape=t.shape))
# #plot_model(model, to_file="firstLayer.png")
# print('Adding 1st Dense?')
# #model.add(Conv2D(32, (3,3), activation='relu'))
# model.add(Dense(3))
# #model.add(Dense(4))
# #model.add(Dense(4))
# #model.add(Flatten())
# model.add(Dense(1))
# print(model.output_shape)
# print('Adding 2nd Dense?')

'''
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(16384,)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='softmax'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))
print(model.shape)
'''

# model = Sequential(
# [
# Dense(32, input_shape = (256, 64, 1),name='Dense-Layer-1'),
# Dense(1, name='output-layer'),
# Activation('relu'),
# #Activation('softmax'),
# ])

a = Input(shape=(256,64))
b = Dense(32)(a)
model = Model(inputs=a,outputs=b)

plot_model(model, show_shapes=True, to_file="firstLayer.png")

model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])
#trainNP = np.matrix(t)
#print(trainNP)
trainLabels = np.array(trainLabels)
trainLabels = np.expand_dims(trainLabels, axis=1)
print('Shape = ',trainLabels.shape)
model.fit(t, trainLabels, epochs=50, batch_size=32)

