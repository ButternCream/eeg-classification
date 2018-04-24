from features_extract import extract_features
from Data import load
from Utils import normalize, concat, create_labels
from sklearn import metrics, svm
import numpy as np


print("Loading")
a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')

print('Created test data')

# create training variables
train = concat(a_train, c_train)
normalize(train)
samples, r,c = train.shape
train = np.reshape(train, (samples,r*c))
train_labels = create_labels(len(a_train), len(c_train)).ravel()


#create and fit SVM classifier
classifier = svm.SVC()
classifier.fit(train,train_labels)
print("Done fitting\nCreating test variables")

#create testing variables
test = concat(a_test, c_test)
normalize(test)
samples, r,c = test.shape
test = np.reshape(test, (samples,r*c))
test_labels = create_labels(len(a_test), len(c_test)).ravel()

print("Starting predictions with test")
results = classifier.predict(test)

print("Accuracy =",metrics.accuracy_score(test_labels, results))
