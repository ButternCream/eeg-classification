from features_extract import extract_features
from Data import load
from sklearn.neighbors import KNeighborsClassifier

print("Loading")
a_train, a_test = load('./alcohol_compressed/', .3)
c_train, c_test = load('./control_compressed/', .3)
print('Loaded')

# Test data
t = np.array(a_test)
#t = np.expand_dims(t,axis=3)

# simpler format for building your arrays
x_one = np.array(a_train)
x_two = np.array(c_train)
x = np.concatenate((x_one, x_two))

y_one = np.ones((x_one.shape[0], 1)) # create array of ones for those in alc
y_two = np.zeros((x_two.shape[0], 1)) # create array of 0's for those in control
y = np.concatenate((y_one, y_two)) # concat in the same order as x
x = np.expand_dims(x, axis=3)
print("x shape: ", x.shape)
print("y shape: ", y.shape)

# normalize x
x = x / 255

classifier = KNeighborsClassifier(n_neighbors=4)
classifier.fit(x,y)

print(classifier.predict(t))
