import os
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM, Embedding
from keras.layers import LocallyConnected2D, GaussianDropout
from keras.utils import np_utils
from Data import fetch_and_compress, load
from keras.utils import plot_model
import matplotlib.pyplot as plt
from Utils import *

model = None
a_train, a_test = load('./alcohol_compressed/', .8)
c_train, c_test = load('./control_compressed/', .8)
if not os.path.isfile("eeg_TrainedModel.h5"):
        
    # simpler format for building your arrays

    train_one = np.array(a_train)
    train_two = np.array(c_train)
    train = np.concatenate((train_one, train_two))
    labels_train = create_labels(len(a_train),len(c_train)).ravel()
    normalize(train)

    test_data = concat(np.array(a_test),np.array(c_test))
    test_data_labels = create_labels(len(a_test), len(c_test)).ravel()
    normalize(test_data)

    #train = convert_to_images(train, 224)
    #test_data = convert_to_images(test_data, 224)
    
    print("Created testing data and labels")

    '''
    # VGG Base model from keras
    base_model = keras.applications.vgg16.VGG16(weights="imagenet", include_top=False,input_shape=(224,224,3), pooling='avg')
    input = Input(shape=(224,224,3))
    output_conv = base_model(input)
    x = Dense(1024, activation='relu', name='layer1')(output_conv)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input, outputs=x)
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
    model.add(LocallyConnected2D(32, 3, activation='relu',input_shape=(224,224,3)))
    model.add(Dense(32, activation='relu'))
    model.add(GaussianDropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    '''

    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.0001) # see keras optimizers for other examples

    model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    print("model compiled")
    
    from keras.callbacks import EarlyStopping
    my_callbacks = [EarlyStopping(monitor='loss', patience=3, mode='auto')]
    
    history = model.fit(train, labels_train, epochs=15, batch_size=32, callbacks=my_callbacks)

    save_plot(history, "original_model.png")
    
    #save model
    print("saving model...")
    model.save("eeg_TrainedModel.h5")

else:
    print("Loading trained model.")
    model = load_model("eeg_TrainedModel.h5")
    
print("Using eval function")
test_data_labels = np.expand_dims(test_data_labels, axis=1)
print(test_data.shape)
print(test_data_labels.shape)
scores = model.evaluate(test_data, test_data_labels, verbose=0)
print("eval acc = ", scores[1]*100)
print("eval size =", len(scores))
print("Testing using predictions...")
preds = model.predict(test_data, batch_size=32, verbose=1)
print(preds)
print(preds.shape)

# Thank you matt <3
cm = confusion_matrix(test_data_labels,preds,labels=[1,0])
print_cm(cm,labels=['Non-Sober', 'Sober'])

