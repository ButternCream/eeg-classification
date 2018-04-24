from Utils import extract_features
from Data import load
from Utils import normalize, concat, create_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')

# simpler format for building your arrays
x = concat(a_train, c_train)
print('concatenated x1 and x2')

samples, r,c = x.shape
x = np.reshape(x, (samples,r*c))

y = create_labels(len(a_train), len(c_train)).ravel()

print("Getting features... come back in 30 minutes.")
a_train_features = np.load("a_train_feat.npy")
print(a_train_features)
a_test_features = np.load("a_test_feat.npy")
c_train_features = np.load("c_train_feat.npy")
c_test_features = np.load("c_test_feat.npy")
print("How was dinner?")

print("Created training data and labels")
print("x shape: ", x.shape)
print("y shape: ", y.shape)

# normalize x
normalize(x)

print("Normalized x")

test_data = concat(a_test,c_test)

samples, r,c = test_data.shape
test_data = np.reshape(test_data, (samples,r*c))

test_data_labels = create_labels(len(a_test), len(c_test)).ravel()

print("Created testing data and labels")

normalize(test_data)

print("Normalized test data")

print()
for d in range(1,15):
    print("Depth = ", d)
    clf = RandomForestClassifier(max_depth=d)
    clf.fit(x,y)

    results = clf.predict(test_data)
    print(metrics.accuracy_score(test_data_labels,results))
