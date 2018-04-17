import numpy as np
from threading import Thread

def normalize(array):
    a_max = np.max(array)
    a_min = np.min(array)
    diff = a_max - a_min
    array = array / diff

def concat(list_one, list_two):
    return np.concatenate((np.array(list_one), np.array(list_two)))

def create_labels(shape_one, shape_two):
    return np.concatenate((np.ones((shape_one, 1)), np.zeros((shape_two,1))))


class Features:
    #expects giant list; main driver
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
            t_zero = Thread(target=__threaded_zeros, args=(A, i, zeros))
            threads.append(t_zero)

            # amount of local maxes and mins
            t_locals = Thread(target=__find_locals, args=(A, i, lmaxes, lmins))
            threads.append(t_locals)
            
            # Mean / Avg
            t_mean = Thread(target=__find_mean, args=(A,i,means))
            threads.append(t_mean)

            # Median
            t_median = Thread(target=__find_median, args=(A,i,medians))
            threads.append(t_median)
            # Could probably shorten everything to
            # threads += [Thread(target=__find_median, args=(A,i,medians)) for i in range(64)]

            # Sum
            t_sums = Thread(target=__find_sums, args=(A,i,sums))
            threads.append(t_sums)
            
            # Need to redo global functions
            # Global min
            t_globalMin = Thread(target=__find_global_min, args=(A,i,gmins))
            threads.append(t_globalMin)
            
            # Global max
            t_globalMax = Thread(target=__find_global_max, args=(A,i,gmaxes))
            threads.append(t_globalMax)
            
        
            
        for j in threads: #start threads
            j.start()
        for k in threads: #join threads; wait for all threads to finish
            k.join()

        features = np.transpose(np.array([zeros, lmaxes, lmins, means, medians, sums, gmaxes, gmins], dtype='float32'))#add other arrays here comma seperated, i.e. [zeros, lmaxes,...]
        return features


    def __threaded_zeros(A, col, zs):
        crossings = 0
        for i in range(1, 256):
            if(A[i-1][col] < 0 and A[i][col] > 0) or (A[i-1][col] > 0 and A[i][col] < 0):
                #^ check for 0 crossing
                crossings+=1
        zs[col] = crossings

    def __find_locals(A, col, maxes_array, mins_array):
        maxes = 0
        mins = 0
        for i in range(1, 255):
            if(A[i-1][col] < A[i][col] and A[i][col] > A[i+1][col] ):
                maxes += 1
            elif(A[i-1][col] > A[i][col] and A[i][col] < A[i+1][col]):
                mins += 1
        maxes_array[col] = maxes
        mins_array[col] = mins
        
    def __find_mean(A, col, means_array):
        means_array[col] = sum(A[:][col])/256

    def __find_median(A,col,medians_array):
        sorted_list = [A[i][col] for i in range(0,256)]
        sorted_list.sort()
        med_index = int(len(sorted_list)/2)
        if len(sorted_list) % 2 == 0:
            medians_array[col] = ((sorted_list[med_index-1] + sorted_list[med_index]) / 2)
        else:
            medians_array[col] = sorted_list[med_index]

    def __find_sums(A,col,sums_array):
        sums_array[col] = sum(A[:][col])


    def __find_global_min(A,col,gmin_array):
        smallest = None
        for i in range(0,256):
            if smallest is None or A[i][col] < smallest:
                smallest = A[i][col]
        gmin_array[col] = smallest

    def __find_global_max(A,col,gmax_array):
        biggest = None
        for i in range(0,256):
            if biggest is None or A[i][col] > biggest:
                biggest = A[i][col]
