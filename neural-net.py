import os
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, Embedding
from keras.layers import LocallyConnected1D, GaussianDropout
from keras.utils import np_utils
from Data import fetch_and_compress, load
from keras.utils import plot_model
import matplotlib.pyplot as plt
from Utils import extract_features
# Compress each persons file data into one numpy file
print("Loading")
a_train, a_test = load('./alcohol_compressed/', .8)
c_train, c_test = load('./control_compressed/', .8)
print('Loaded')

print("Extracting features")
features = extract_features(c_test)

model = None
if not os.path.isfile("eeg_TrainedModel.h5"):
    # simpler format for building your arrays
    train_one = np.array(a_train)
    train_two = np.array(c_train)
    train = np.concatenate((train_one, train_two))
    
    '''
    train = train.flatten()[:100000]
    x = np.array([i for i in range(len(train))])
    
    print(x.shape)
    print(train.shape)

    plt.scatter(x, train, alpha=0.5)
    plt.show()
    '''

    labels_tr_one = np.ones((train_one.shape[0], 1)) # create array of ones for those in alc
    labels_tr_two = np.zeros((train_two.shape[0], 1)) # create array of 0's for those in control
    labels_train = np.concatenate((labels_tr_one, labels_tr_two)) # concat in the same order as x
    #train = np.expand_dims(train, axis=3)
    print("train shape: ", train.shape)
    print("labels_train shape: ", labels_train.shape)
    
    # normalize x
    train = train / np.amax(train)
    
# build our model. The example provided by Boaz/Mathew requires a base model, this builds a model without a base provided. 
    '''
    # Original
    model = Sequential()
    model.add(Dense(256, input_shape=train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten()) # must flatten to ensure we have a 1d array going into our prediction layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    '''
    # Custom CNN
    model = Sequential()
    model.add(LocallyConnected1D(32, 3, activation='relu',input_shape=train.shape[1:]))
    model.add(Dense(32, activation='relu'))
    model.add(GaussianDropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    '''
    # VGG-Like CNN
    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=train.shape[1:]))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    '''
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001) # see keras optimizers for other examples

    model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    print("model compiled")
    
    from keras.callbacks import EarlyStopping
    my_callbacks = [EarlyStopping(monitor='loss', patience=5, mode='auto')]
    
    model.fit(train, labels_train, epochs=15, batch_size=32, callbacks=my_callbacks)
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
#test = np.expand_dims(test, axis=3)

labels_t_one = np.ones((test_one.shape[0], 1)) # create array of ones for those in alc
labels_t_two = np.zeros((test_two.shape[0], 1)) # create array of 0's for those in control
labels_test = np.concatenate((labels_t_one, labels_t_two)) # concat in the same order as x

# normalize test
test = test / np.amax(test)

print("Using eval function")
scores = model.evaluate(test, labels_test, verbose=0)
print("eval acc = ", scores[1]*100)
print("eval size =", len(scores))
print(scores)

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

