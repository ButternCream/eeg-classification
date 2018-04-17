import os
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
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

model = None
if not os.path.isfile("eeg_TrainedModel.h5"):
    # simpler format for building your arrays
    train_one = np.array(a_train)
    train_two = np.array(c_train)
    train = np.concatenate((train_one, train_two))
    
    
    labels_tr_one = np.ones((train_one.shape[0], 1)) # create array of ones for those in alc
    labels_tr_two = np.zeros((train_two.shape[0], 1)) # create array of 0's for those in control
    labels_train = np.concatenate((labels_tr_one, labels_tr_two)) # concat in the same order as x
    train = np.expand_dims(train, axis=3)
    print("train shape: ", train.shape)
    print("labels_train shape: ", labels_train.shape)
    
    # normalize x
    train = train / 255
    
# build our model. The example provided by Boaz/Mathew requires a base model, this builds a model without a base provided. 
    model = Sequential()
    model.add(Dense(64, input_shape=train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(64))
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
    
    model.fit(train, labels_train, epochs=9, batch_size=32, callbacks=my_callbacks)
    #save model
    print("saving model...")
    model.save("eeg_TrainedModel.h5")

else:
    print("Loading trained model.")
    model = load_model("eeg_TrainedModel.h5")
    

# simpler format for building your arrays
test_one = np.array(a_test)
test_two = np.array(c_test)
test = np.concatenate((test_one, test_two))


labels_t_one = np.ones((test_one.shape[0], 1)) # create array of ones for those in alc
labels_t_two = np.zeros((test_two.shape[0], 1)) # create array of 0's for those in control
labels_test = np.concatenate((labels_t_one, labels_t_two)) # concat in the same order as x
test = np.expand_dims(test, axis=3)

# normalize test
test = test / 255
print("Using eval function")
scores = model.evaluate(test, labels_test, verbose=0)
print("eval acc = ", scores[1]*100)
print("eval size =", len(scores))

print("Testing using predictions...")
preds = model.predict(test, batch_size=32, verbose=1)

rounded = [round(x[0]) for x in preds]

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(labels_t_one)):
    if(rounded[i] == labels_t_one[i]):
        tp+=1
    else:
        fp+=1
len_ones = len(labels_t_one)
for i in range(len(labels_t_two)):
    if(rounded[i+len_ones] == labels_t_two[i]):
        tn+=1
    else:
        fn+=1

print("pred acc =", (tp+tn)/len(test))

    
#print(rounded)

