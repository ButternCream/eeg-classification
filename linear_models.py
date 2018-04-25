from Utils import extract_features
from Data import load
from Utils import normalize, concat, create_labels
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

names = ["Stochastic Gradient Descent", "Passive Aggressive", "Ridge", "Perceptron", "SVC"]
models = [linear_model.SGDClassifier(), linear_model.PassiveAggressiveClassifier(), linear_model.RidgeClassifier(), 
        linear_model.Perceptron()]# SVC()]
scores = [[], [], [], []]
for _ in range(100):
    for i, clf in enumerate(models):
        print("Current: ", names[i])
        clf.fit(x,y)
        results = clf.predict(test_data)
        s = metrics.accuracy_score(test_data_labels, results)
        scores[i].append(s)
        print("Accuracy: ", s)

SGD_avg = np.average(np.array(scores[0]))
PA_avg = np.average(np.array(scores[1]))
Ridge_avg = np.average(np.array(scores[2]))
Perceptron_avg = np.average(np.array(scores[3]))
print("Averages") 
print("SGD: ", SGD_avg)
print("PA: ", PA_avg)
print("Ridge: ", Ridge_avg)
print("Perceptron: ", Perceptron_avg)
