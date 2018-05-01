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

        
# simpler format for building your arrays

train_one = np.array(a_train[5])
train_two = np.array(c_train[5])
train = np.concatenate((train_one, train_two))
labels_train = create_labels(len(a_train),len(c_train)).ravel()
normalize(train)

test_data = concat(np.array(a_test[0:5]),np.array(c_test[0:5])) # Naz
test_data_labels = create_labels(len(a_test), len(c_test)).ravel()
normalize(test_data)

test_data = convert_to_images(test_data, 224)
plt.imshow(test_data[0])
plt.savefig('eeg_alcoholic_image.png')
plt.imshow(test_data[8]) # Naz
plt.savefig('eeg_control_image.png')

