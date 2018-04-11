from features_extract import extract_features
from Data import load

#load control for testing
def get_features():
    c_train, c_test = load('./control_compressed', .3)
    print("len(c_train):", len(c_train))
    print("len(c_test):", len(c_test))
    ################Testing extract_features#################
    feats = extract_features(c_train[-3:])
    print("length c_train[:-3]", len(c_train[-3:]))
    print("printing FP1 column:")
    for i in c_train[-3]:
        print(i[0])
    print("printing features for the above columnn")
    print(feats[0][0])
    #########################################################


get_features()
