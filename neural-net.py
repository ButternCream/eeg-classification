import os
import numpy as np
import keras
from Utils import concat, normalize, create_labels
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
    print("Model not created. Creating neural net model...")
    # create training variables                                                                                    
    train = concat(a_train, c_train)
    train = train/255
    samples, r,c = train.shape
    train = np.expand_dims(train, axis=3)
    train_labels = create_labels(len(a_train), len(c_train)).ravel()
    

    # build our model. The example provided by Boaz/Mathew requires a base model, this builds a model without a base provided. 
    print("Initializing model")
    model = Sequential()
    model.add(Dense(8, input_shape=train.shape[1:]))
    model.add(Activation('relu'))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    model.add(Flatten()) # must flatten to ensure we have a 1d array going into our prediction layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    print("Compiling model")
    optimizer = keras.optimizers.Adam(lr=0.001) # see keras optimizers for other examples
    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    print("model compiled")
    
    from keras.callbacks import EarlyStopping
    my_callbacks = [EarlyStopping(monitor='acc', patience=5,mode=max)]
    
    model.fit(train, train_labels, epochs=9, batch_size=32, callbacks=my_callbacks)
    #save model
    print("Saving model")
    model.save("eeg_TrainedModel.h5")

else:
    print("Loading trained model.")
    model = load_model("eeg_TrainedModel.h5")
    
#create testing variables                                                                                      
test = concat(a_test, c_test)/255
samples, r,c = test.shape
test = np.expand_dims(test, axis=3)
test_labels = create_labels(len(a_test), len(c_test)).ravel()

print("Using eval function")
scores = model.evaluate(test, test_labels, verbose=0)
print("eval acc = ", scores[1]*100)

print("Testing using predictions...")
preds = model.predict(test, batch_size=32, verbose=1)

rounded = [round(x[0]) for x in preds]

tp = 0
fp = 0
tn = 0
fn = 0
for i in range(len(a_test)):
    if(rounded[i] == test_labels[i]):
        tp+=1
    else:
        fp+=1
len_ones = len(a_test)
for i in range(len(c_test)):
    if(rounded[i+len_ones] == test_labels[i+len_ones]):
        tn+=1
    else:
        fn+=1

print("pred acc =", (tp+tn)/len(test))


