import numpy as np

def normalize(array):
    a_max = np.max(array)
    a_min = np.min(array)
    diff = a_max - a_min
    array = array / diff

def concat(list_one, list_two):
    return np.concatenate((np.array(list_one), np.array(list_two)))

def create_labels(shape_one, shape_two):
    return np.concatenate((np.ones((shape_one, 1)), np.zeros((shape_two,1))))
