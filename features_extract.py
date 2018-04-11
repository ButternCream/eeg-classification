from threading import Thread
#expects giant list; main driver
def extract_features(L):
    features = []
    for i in L:
        features.append(extract_for_file(i))
        
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
    ##8. global max (check the max of 63 others and 1 for max and 0 for other 63)
    gmaxes = [[]for i in range(64)]
    threads = []
    for i in range(64): #create threads
        z = Thread(target=threaded_zeros, args=(A, i, zeros))
        threads.append(z)
        

    for j in threads: #start threads
        j.start()
    for k in threads: #join threads; wait for all threads to finish
        k.join()
    
def threaded_zeros(A, col_idx, zs):
    crossings = 0;
    for i in range(1, 256):
        if(A[i-1][col_idx] < 0 && A[i][col_idx] > 0) || (A[i-1][col_idx] > 0 && A[i][col_idx] < 0):
            #^ check for 0 crossing
            crossings+=1
    zs[col_idx] = crossings
    
    
