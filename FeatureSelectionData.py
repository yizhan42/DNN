from __future__ import print_function, division
from sklearn.feature_selection import VarianceThreshold
# from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import KFold

import csv
import pandas as pd
import numpy as np

import os

from main import *
from process_data import *

# select total_num samples to test
def All_Data(read_csv_file, write_csv_file, total_num, num=0 ):
    data = open(write_csv_file,'w')

    with open(read_csv_file) as fp:
        for line in fp:
            if num < total_num:
                data.write(line)
                num += 1


def merge_data(read_csv_file1, read_csv_file2, write_csv_file):
    merge = open(write_csv_file, 'w')
    with open(read_csv_file1, 'r') as fp:
        fp.readline()  # read a line
        for line in fp:
            merge.write(line)
    with open(read_csv_file2,'r') as fq:
        fq.readline()
        for line in fq:
            merge.write(line)
    merge.close()

# merge previous train data and validation data to get a new all data in order to feature selection, then split them
# pay attention to "no fp.readline()" here
def merge_fs_data(read_csv_file1, read_csv_file2, write_csv_file):
    merge = open(write_csv_file, 'w')
    with open(read_csv_file1, 'r') as fp:
        for line in fp:
            merge.write(line)
    with open(read_csv_file2,'r') as fq:
        for line in fq:
            merge.write(line)
    merge.close()

# make all data to list
def merge_list(read_csv_file):
    fs_all_list = []

    with open(read_csv_file, 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            fs_all_list.append(line[1:])

    print(len(fs_all_list))
    fs_all_list = pd.DataFrame(data=fs_all_list)
    print(fs_all_list.shape)
    print(fs_all_list[1][1])
    print(type(fs_all_list[1][1]))
    return fs_all_list

# feature selection all data
# Removing features with low variance
# VarianceThreshold is a simple baseline approach to feature selection. 
# It removes all features whose variance doesn’t meet some threshold. 
# By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

# As an example, suppose that we have a dataset with boolean features, 
# and we want to remove all features that are either one or zero (on or off) in more than 80% of the samples. 
# Boolean features are Bernoulli random variables, and the variance of such variables is given by

# Var[X] = p(1 - p)

# so we can select using the threshold .8 * (1 - .8):
# >>> from sklearn.feature_selection import VarianceThreshold
# >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
# >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# >>> sel.fit_transform(X)
# array([[0, 1],
#        [1, 0],
#        [0, 0],
#        [1, 1],
#        [1, 0],
#        [1, 1]])
# As expected, VarianceThreshold has removed the first column, which has a probability p = 5/6 > .8 of containing a zero.


def fs_merge_list(fs_list, write_csv_file):
    sel = VarianceThreshold(threshold=(.001 * (1 - .001)))
    after_fs_all_list = sel.fit_transform(fs_list)
    # print ("&&&&")
    # print (len(after_fs_all_list))
    with open(write_csv_file,'w') as fp:
        fp_csv = csv.writer(fp)
        fp_csv.writerows(after_fs_all_list)
    # print(len(after_fs_all_list))
    # print('***')
    # print(after_fs_all_list.shape)


def crossValidationSplit(read_csv_file, save_folder, filename_prefix = 'part', n_splits = 10):
    '''Split data into two part: traning and validation, for cross validating.

    Arguments:
        data_path: the path of the data file.
        save_folder: the folder path to save data.
        filename_prefix: the prefix name of each save file.
        n_splits: the number of parts to split the data into.
        transfer: a transfer function to change the sequence format.
        **args: the args of transfer function. You can find them in sepific transfer function.
    '''
    with open(read_csv_file, 'r') as rf:
        size = len(rf.readlines())
        rf.seek(0, 0)
       
        reader = pd.read_csv(rf)
        # Create save folder if it is not existed
        if(not os.path.exists(save_folder)):
            os.makedirs(save_folder)

        all_list = random.sample(range(0, size), size) # mix all data
        interval = size // n_splits
        save_writers = list()
        print('size:', size, 'interval:', interval)
        for i in range(n_splits):
            wf = open('{}/{}_{}.csv'.format(save_folder, filename_prefix, i), 'w')
            # save_writers.append(csv.writer(wf, delimiter=',', quotechar='|'))
            save_writers.append(csv.writer(wf)) #save_writers is a list, its elelments are csv.writer(wf)
       
        X = reader.iloc[:, 0:].values
        for index, item in enumerate(X):
            no = all_list.index(index) // interval
            if no == n_splits:
                no = n_splits - 1
            save_writers[no].writerow(item) #save_writers[no] is csv.writer(wf)
            # save_writers[no].append()




# split new all data(which had been feature selected) to train and validation
def split(read_csv_file, write_csv_file1, write_csv_file2, model):
    df_biology = pd.read_csv(read_csv_file)
    # the code below is important to the shape of train and validation
    X = df_biology.iloc[:, 0:].values
   
    print (X[0])
   
    kf = KFold(n_splits=10, shuffle=False)
    for train_index, val_index in kf.split(X):
        # print('train_index:%s , test_index: %s ' % (train_index, val_index))
        train, validation = X[train_index], X[val_index]
        # y_train, y_test = y[train_index], y[test_index]

    print ("$$$$")
    print (len(train))


    print("____")
    print(len(train))
    print(len(validation))
    print (train[0][0])

    # count number of label = 1.0
    length = len(train)
    count = 0
    # i = 0
    for i in range(length):
        if train[i][0] == 1:
            count += 1
        i += 1
    print (count)
   
    with open(write_csv_file1, 'w') as fp:
        f_csv = csv.writer(fp)
        f_csv.writerows(train)

    with open(write_csv_file2, 'w') as fq:
        f_csv = csv.writer(fq)
        f_csv.writerows(validation)


def fs_merge_train_list():
    fs_train_list = []
    with open('after_fs_new_train_formal.csv', 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            fs_train_list.append(line)

    fs_train_list = pd.DataFrame(data=fs_train_list)

    print(len(fs_train_list))
    print('***')
    print(fs_train_list.shape)
    print(fs_train_list[0][0])
    print(type(fs_train_list[1][1]))
    return fs_train_list

def fs_merge_val_list():
    fs_val_list = []
    with open('after_fs_new_val_formal.csv', 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            fs_val_list.append(line)

        fs_val_list = pd.DataFrame(data=fs_val_list)

    return fs_val_list




# after feature selection, split train(validation) and test, train and validation are together,
# these are final data
def randomSplit_trainval_test(csv_file, size, pick_rate):
    test_list = random.sample(range(1, size + 1), int(pick_rate * size))
    i = 1

    train_val = open('final_train_val.csv', 'w')
    testf = open('final_test.csv', 'w')
    with open(csv_file, 'r') as fp:
        for line in fp:
            # line = line.replace('\r\n','').split(',')
            if i in test_list:
                testf.write(line)
            else:
                train_val.write(line)

            i += 1

    train_val.close()
    testf.close()

if __name__ == '__main__':
    

    # the following is for real
    merge_data('pos_train_new.csv', 'neg_train.csv', 'fs_train_all_formal.csv')
    merge_data('pos_val.csv', 'neg_val.csv', 'fs_val_all_formal.csv')
    merge_data('pos_test.csv', 'neg_test.csv', 'fs_all_test_formal.csv')
    merge_fs_data('fs_train_all_formal.csv', 'fs_val_all_formal.csv', 'fs_all_training_formal.csv')
    merge_fs_data('fs_all_training_formal.csv', 'fs_all_test_formal.csv', 'fs_all.csv')


    train_val_test_list = merge_list('fs_all.csv')
    fs_merge_list(train_val_test_list, 'after_fs_all_train_val_test_data_formal.csv')
    randomSplit_trainval_test('after_fs_all_train_val_test_data_formal.csv', 3812, 0.09)
    model = TwoLayerNet(D_in, H, I, J, K, D_out)
    # split('after_fs_all_data_formal.csv', 'after_fs_new_train_formal.csv', 'after_fs_new_val_formal.csv',model)
    crossValidationSplit('final_train_val.csv','/home/chunhui/yi/DNN/NewData/data/train')
    

    test_list = merge_list('fs_all_test_formal.csv')
    fs_merge_list(test_list,'after_fs_all_test_data_formal.csv')
    crossValidationSplit('final_test.csv', '/home/chunhui/yi/DNN/NewData/data/test')
    