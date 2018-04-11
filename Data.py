import os
import numpy as np
import time

def timer(func):
    def f(*args, **kwargs):
        before = time.time()
        return_func = func(*args, **kwargs)
        after = time.time()
        print('elapsed',after-before)
        return return_func
    return f

def compress_person_data(file_list):
    #values = [[] for l in range(len(file_list))]
    values = []
    
    for i,filename in enumerate(file_list):#loop through files
        with open(filename) as myfile:
            fileVals = [[] for l in range(256)]
            for k,line in enumerate(myfile):#loop through lines
                if '#' not in line:
                    s = line.split()
                    fileVals[k%256].append(float(s[3]))
                    #values[(i*256) + (k%256)].append(float(s[3]))
                    #values.append(float(s[3])
            values.append(fileVals)
    return values

def construct_file_list(root_path, files):
    path_list = []
    for f in files:
        if '.rd.' in f:
            file_path = os.path.join(root_path, f)
            path_list.append(file_path)
    return path_list
    

def compress_and_save(label):
    total = 0
    if 'alc' in label:
        save_dir = './alcohol_compressed/'
    elif 'cont' in label:
        save_dir = './control_compressed/'
    else:
        save_dir = ''
    print("Current: " + label)
    for dirpath, dirs, files in os.walk(label):
        path = dirpath
        f = files
        np_filename = label.split('/')[1][:3] + '_person_' + str(total)
        if not os.path.isfile(save_dir+np_filename+'.npy'):
            file_paths = construct_file_list(path, f)
            values = compress_person_data(file_paths)
            print("Person: " + str(total) + " " + str(len(values)))
            if len(values) > 0:
                np.save(save_dir+np_filename, values)
        total += 1

def fetch_and_compress(labels):
    for label in labels:
        compress_and_save(label)
#@timer
def load(label, percentage, testing=False):
    train = []
    test = []
    for dirpath, dirs, files in os.walk(label):
        total  = int(percentage*len(files))
        print("Getting " + str(total) + "/" + str(len(files)) + " files")
        for i,f in enumerate(sorted(files)):
            path = os.path.join(dirpath, f)
            print(path)
            if i < total:
                temp = np.load(path, fix_imports=True).tolist()
                if len(np.array(temp).shape) != 2:
                    for l in temp:
                        train.append(l)
                #train.append(np.load(path).tolist())
            else:
                temp = np.load(path, fix_imports=True).tolist()
                if len(np.array(temp).shape) != 2:
                    for l in temp:
                        test.append(l)
                #test.append(np.load(path).tolist())
    return train, test

load('./control_compressed/', .7, testing=True)
