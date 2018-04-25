from Utils import extract_features
from Data import load
from Utils import normalize, concat, create_labels
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import metrics
import numpy as np

a_train, a_test = load('./alcohol_compressed/', .7)
c_train, c_test = load('./control_compressed/', .7)
print('Loaded')

# simpler format for building your arrays
x = concat(a_train, c_train)
print('concatenated x1 and x2')

print(x.shape)
samples, r,c = x.shape
x = np.reshape(x, (samples,r*c))
print(x.shape)

y = create_labels(len(a_train), len(c_train)).ravel()
print(y)

# normalize x
#normalize(x)
x = x / 1024

test_data = concat(a_test,c_test)

samples, r,c = test_data.shape
test_data = np.reshape(test_data, (samples,r*c))

test_data_labels = create_labels(len(a_test), len(c_test)).ravel()

#normalize(test_data)
test_data = test_data / 1024

names = ["Stochastic Gradient Descent", "Passive Aggressive", "Ridge", "Perceptron"]
models = [linear_model.SGDClassifier(), linear_model.PassiveAggressiveClassifier(), linear_model.RidgeClassifier(), linear_model.Perceptron()]
for i, clf in enumerate(models):
    print("Current: ", names[i])
    clf.fit(x,y)
    results = clf.predict(test_data)
    print("Accuracy: ", metrics.accuracy_score(test_data_labels, results))

