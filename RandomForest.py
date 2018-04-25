from Utils import extract_features
from Data import load
from Utils import normalize, concat, create_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')

a_train = np.array(a_train)

print(a_train.shape)

print("Getting features... come back in 30 minutes.")
a_train_features = np.load("a_train_feat.npy")
a_test_features = np.load("a_test_feat.npy")
c_train_features = np.load("c_train_feat.npy")
c_test_features = np.load("c_test_feat.npy")
print("How was dinner?")

# simpler format for building your arrays
x = concat(a_train_features, c_train_features)
print('concatenated x1 and x2')

print(x.shape)
samples, r,c = x.shape
x = np.reshape(x, (samples,r*c))
print(x.shape)

y = create_labels(len(a_train_features), len(c_train_features)).ravel()
print(y)

# normalize x
#x = x / np.amax(x)

print("Normalized x")

test_data = concat(a_test_features,c_test_features)

samples, r,c = test_data.shape
test_data = np.reshape(test_data, (samples,r*c))

test_data_labels = create_labels(len(a_test_features), len(c_test_features)).ravel()

print("Created testing data and labels")

#test_data = test_data / np.amax(test_data)

print("Normalized test data")

print()
# Depth of 7 seems to work well
for d in range(1,15):
    print("Depth = ", d)
    clf = RandomForestClassifier(max_depth=d)
    clf.fit(x,y)

    results = clf.predict(test_data)
    print(metrics.accuracy_score(test_data_labels,results))
