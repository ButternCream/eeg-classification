from features_extract import extract_features
from Data import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

print("Loading")
a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')

# Test data
t = np.array(a_test)

print('Created test data')

# simpler format for building your arrays
x_one = np.array(a_train)
x_two = np.array(c_train)
x = np.concatenate((x_one, x_two))

print('concatenated x1 and x2')

samples, r,c = x.shape
x = np.reshape(x, (samples,r*c))
samples, r,c = t.shape
t = np.reshape(t, (samples,r*c))

y_one = np.ones((x_one.shape[0], 1)) # create array of ones for those in alc
y_two = np.zeros((x_two.shape[0], 1)) # create array of 0's for those in control
y = np.concatenate((y_one, y_two)) # concat in the same order as x
#y = np.ones((t.shape[0], 1))
# x = np.expand_dims(x, axis=3)
print("x shape: ", x.shape)
print("y shape: ", y.shape)
print("Length of t: ", len(t))
print("t: ", t[:100])

# normalize x
x = x / 255
y = y.ravel()

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(x,y)
results = classifier.predict(x)
print(metrics.accuracy_score(y, results))

# results = classifier.predict(t) # Doesn't work
# print(metrics.accuracy_score(t, results)) # ^



