import numpy as np
from threading import Thread
from Data import timer
from PIL import Image
import matplotlib.pyplot as plt

def save_plot(history, filename):
    plt.plot(history.history['acc'])
    plt.plot(history.history['loss'])
    plt.title('Model Accuracy vs Loss')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper right')
    #plt.show()
    plt.savefig(filename)

def normalize(array):
    a_min = np.amin(array)
    array = array - a_min
    array = array / 255

def concat(list_one, list_two):
    return np.concatenate((np.array(list_one), np.array(list_two)))

def create_labels(shape_one, shape_two):
    return np.concatenate((np.ones((shape_one, 1)), np.zeros((shape_two,1))))


@timer
def extract_features(L):
    features = []
    for i in L:
        features.append(extract_for_file(i))
    return features

#expects 256x64 array
def extract_for_file(A):
    #feature will be the following:
    ##1. number of zero crossings
    zeros = [[]for i in range(64)] 
    ##2. #of local maxes
    lmaxes = [[]for i in range(64)] 
    ##3. #of local mins
    lmins = [[]for i in range(64)]
    ##4. average/mean
    means = [[]for i in range(64)]
    ##5. median
    medians = [[]for i in range(64)]
    ##6. sum
    sums = [[]for i in range(64)]
    ##7. global min
    gmins = [[]for i in range(64)]
    ##8. global max (check the max of 63 others; 1 for max and 0 for other 63)
    gmaxes = [[]for i in range(64)]
    threads = []
    for i in range(64): #create threads
        #zero crossings
        t_zero = Thread(target=threaded_zeros, args=(A, i, zeros))
        threads.append(t_zero)

        # amount of local maxes and mins
        t_locals = Thread(target=find_locals, args=(A, i, lmaxes, lmins))
        threads.append(t_locals)
        
        # Mean / Avg
        t_mean = Thread(target=find_mean, args=(A,i,means))
        threads.append(t_mean)

        # Median
        t_median = Thread(target=find_median, args=(A,i,medians))
        threads.append(t_median)
        # Could probably shorten everything to
        # threads += [Thread(target=find_median, args=(A,i,medians)) for i in range(64)]

        # Sum
        t_sums = Thread(target=find_sums, args=(A,i,sums))
        threads.append(t_sums)
        
        # Need to redo global functions
        # Global min
        t_globalMin = Thread(target=find_global_min, args=(A,i,gmins))
        threads.append(t_globalMin)
        
        # Global max
        t_globalMax = Thread(target=find_global_max, args=(A,i,gmaxes))
        threads.append(t_globalMax)
        
    
        
    [j.start() for j in threads] # Start threads
    [k.join() for k in threads] # Wait for them to finish

    features = np.transpose(np.array([zeros, lmaxes, lmins, means, medians, sums, gmaxes, gmins])) #add other arrays here comma seperated, i.e. [zeros, lmaxes,...]
    return features


def threaded_zeros(A, col, zs):
    crossings = 0
    for i in range(1, 256):
        if(A[i-1][col] < 0 and A[i][col] > 0) or (A[i-1][col] > 0 and A[i][col] < 0):
            #^ check for 0 crossing
            crossings+=1
    zs[col] = crossings

def find_locals(A, col, maxes_array, mins_array):
    maxes = 0
    mins = 0
    for i in range(1, 255):
        if(A[i-1][col] < A[i][col] and A[i][col] > A[i+1][col] ):
            maxes += 1
        elif(A[i-1][col] > A[i][col] and A[i][col] < A[i+1][col]):
            mins += 1
    maxes_array[col] = maxes
    mins_array[col] = mins
    
def find_mean(A, col, means_array):
    means_array[col] = sum(A[:][col])/256

def find_median(A,col,medians_array):
    sorted_list = [A[i][col] for i in range(0,256)]
    sorted_list.sort()
    med_index = int(len(sorted_list)/2)
    if len(sorted_list) % 2 == 0:
        medians_array[col] = ((sorted_list[med_index-1] + sorted_list[med_index]) / 2)
    else:
        medians_array[col] = sorted_list[med_index]

def find_sums(A,col,sums_array):
    sums_array[col] = sum(A[:][col])


def find_global_min(A,col,gmin_array):
    smallest = None
    for i in range(0,256):
        if smallest is None or A[i][col] < smallest:
            smallest = A[i][col]
    gmin_array[col] = smallest

def find_global_max(A,col,gmax_array):
    biggest = None
    for i in range(0,256):
        if biggest is None or A[i][col] > biggest:
            biggest = A[i][col]
    gmax_array[col] = biggest




def print_cm(cm, labels, hide_zeroes=False,
             hide_diagonal=False, hide_threshold=None, output_file=None):
        """pretty print for confusion matrixes"""

        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = string = StringIO()
        
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
                print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        # Print rows
        for i, label1 in enumerate(labels):
                print("    %{0}s".format(columnwidth) % label1, end=" ")
                for j in range(len(labels)):
                        cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                        if hide_zeroes:
                                cell = cell if float(cm[i, j]) != 0 else empty_cell
                        if hide_diagonal:
                                cell = cell if i != j else empty_cell
                        if hide_threshold:
                                cell = cell if cm[i, j] > hide_threshold else empty_cell
                        print(cell, end=" ")
                print()
                
        sys.stdout = old_stdout
        print(string.getvalue())
        return string.getvalue()

                
def confusion_matrix(Y_true, Y_pred, labels=None, verbose=False):
        '''
        returns array, shape
        '''
        from sklearn.metrics import confusion_matrix

        # by defaults keras uses the round function to assign classes.
        # Thus the default threshold is 0.5

        Y_pred = np.rint(Y_pred[:,0])
        Y_true = Y_true[:,0]
        
        if verbose is True:
                print("Y_true: ", str(Y_true))
                print("Y_preds: ", str(Y_pred))

        return confusion_matrix(Y_true,Y_pred, labels=labels)
        

def convert_to_images(images, image_size):
    new_images = []
    i = 0
    for image in images:
        if image is not None:
            i += 1
            image = np.squeeze(image)
            bw_image = Image.fromarray(image,"L")
            rbg_image = Image.new("RGB", bw_image.size)
            rbg_image.paste(bw_image)
            rbg_image = rbg_image.resize((image_size,image_size), Image.ANTIALIAS)
            np_image = np.array(rbg_image)
            new_images.append(np_image)
    return np.array(new_images)
