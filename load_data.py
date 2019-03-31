# coding=utf8


from __future__ import print_function, division
import os
import pandas as pd
# from skimage import io, transform
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import random
import csv

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from settings import *

# Load Data with testing and training data randomly splited.
# class ProteinDataset(Dataset):
#     """ Proteint Dataset. """
#     def __init__(self, pd_dataFrame, transform = None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         # read csv file without header row
#         self.properties_frame = pd_dataFrame
#         # self.properties_frame = pd.read_csv(csv_file, header = None)
#         # extract different part of data - name, label and properties
#         self.labels = self.properties_frame.ix[:,0].values.astype('float64')
#         # self.labels[self.labels==-1] = 0 # modify -1 to 0
#         # self.protein_names = self.properties_frame.ix[:,0]
#         self.properties_frame = self.properties_frame.ix[:,1:]
#         self.transform = transform

#     def __len__(self):
#         return len(self.properties_frame)

#     def __getitem__(self, idx):
#         # convert pd.DataFrame to np.ndarray or other
#         # protein_name = self.protein_names.ix[idx,:]
#         properties = self.properties_frame.ix[idx,:].values.astype('float')
#         label = self.labels[idx]

#         # properties.resize((WIDTH, HEIGHT))
#         sample = (properties, label)

#         if self.transform:
#             sample = self.transform(sample)
#         return sample
        
#     def get_class_weight(self):
#         num_samples = self.__len__()
#         label_set = set(self.labels)
#         num_class = [self.labels.count(c) for c in label_set]
#         class_weight = [num_samples/float(self.labels.count(c))
#                         for c in label_set]
#         return class_weight, num_class

# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __init__(self, args):
#         self.args = args

#     def __call__(self, sample):
#         properties, label = sample

#         # convert np.ndarray to tensor
#         properties = torch.from_numpy(properties)
#         # insert depth   
#         properties = properties.float().view(1,self.args.length)
#         # properties = properties.float()
#         # properties = properties.float().view(1, HEIGHT, WIDTH)
#         return properties, label


class ProteinDataset(Dataset):
    """ Proteint Dataset. """
    def __init__(self, pd_dataFrame, transform = None):
    
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # read csv file without header row
        self.properties_frame = pd_dataFrame
        # self.properties_frame = pd.read_csv(csv_file, header = None)
        # extract different part of data - name, label and

        # previous data with id (which is the first column)
        # self.labels = self.properties_frame.ix[:,1].as_matrix().astype('int64')

        # after feature selection, data without id(which is the first column before)
        # self.labels = self.properties_frame.ix[:,0].as_matrix().astype('float64')
        self.labels = self.properties_frame.ix[:,0].values.astype('float64')
        # print(self.labels)
        # self.labels[self.labels==-1] = 0 # modify -1 to 0

        # after feature selection, data without protein id, i.e. protein name
        # self.protein_names = self.properties_frame.ix[:,0]
        self.properties_frame = self.properties_frame.ix[:,1:]
        self.transform = transform

    def __len__(self):
        return len(self.properties_frame)

    def __getitem__(self, idx):
        # convert pd.DataFrame to np.ndarray or other

        # after featuer selection,new data without protein_name
        # protein_name = self.protein_names.ix[idx,:]

#print(self.properties_frame.ix[idx, :])
        # properties = self.properties_frame.ix[idx, :].as_matrix().astype('double')

        # properties = self.properties_frame.ix[idx,:].as_matrix().astype('float')
        properties = self.properties_frame.ix[idx,:].values.astype('float')
        label = self.labels[idx]

        # properties.resize((WIDTH, HEIGHT))
        sample = (properties, label)

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, args):
        pass
    def __call__(self, sample):
        properties, label = sample

        # convert np.ndarray to tensor
        properties = torch.from_numpy(properties)
        # insert depth
        properties = properties.float()#.view(1, WIDTH, HEIGHT)

        return properties, label



# Split data for training and testing randomly
def randomSplit(csv_file, pos_size, neg_size, pick_rate):
    test_list = random.sample(range(1, neg_size+1), int(pick_rate * neg_size)) +\
        random.sample(range(neg_size+1, neg_size+pos_size+1), int(pick_rate * pos_size))
    i = 1
    test = []
    train = []
    with open(csv_file, 'r') as fp:
        for line in fp:
            line = line.replace('\n','').split(',')
            if i in test_list:
                test.append(line)
            else:
                train.append(line)
            i += 1
    test = pd.DataFrame(data = test)
    train = pd.DataFrame(data = train)
    lg("Train Shape = {}, Test Shape = {}".format(train.shape, test.shape))
    lg("Test data No = {}\n".format(test_list))
    return train,test


# split train and test
def splitData(X, y):
    # X = array([[0,1],[2,3],[4,5]]), X_train = array([[0,1],[2,3]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)
    return X_train, X_test, y_train, y_test

# split train data to 10 fold cross validation
def kfold(X, y):
    kf = KFold(n_splits=10, shuffle=True)
    # kf.get_n_splits(X)
    for train_index, val_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", val_index)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    return X_train, X_val, y_train, y_val


# def readTrainingData(label_data_path='/home/chunhui/yi/DNN/NewData/data/train/part', index=0, total=10):
def readTrainingData(label_data_path='./data/multihot_data/train/part', index=0, total=10):
    # train_label = []
    train_data = []

    # validation_label = []
    validation_data = []

    for i in range(total):
        with open('{}_{}.csv'.format(label_data_path, i)) as read_file:
            reader = csv.reader(read_file)
            if i == index:
                for item in reader:
                    validation_data.append(item)
                    # validation_label.append(int(label))
            else:
                for item in reader:
                    train_data.append(item)
                    # train_label.append(int(label))
    train_data = pd.DataFrame(data=train_data)
    validation_data = pd.DataFrame(data=validation_data)
    # return ProteinDataset(properties_frame, labels, properties_frame), ProteinDataset(properties_frame, labels, properties_frame)
    # print(validation_data)
    return train_data, validation_data

# def readTestData(label_data_path='/home/chunhui/yi/DNN/NewData/data/test/part', index=0, total=10):
def readTestData(label_data_path='./data/multihot_data/test/test_joint_without_id.csv'):  
    test_data = []   
    with open(label_data_path) as read_file:
        reader = csv.reader(read_file)            
        for item in reader:
            test_data.append(item)

    test_data = pd.DataFrame(data = test_data)
    # test_small_data = pd.DataFrame(data = test_small_data)
    return test_data



def dropFirstColumn():
    for i in range(10):
        with open('data/knnscore_data_0330/train/part_{}'.format(i),'r') as reader, open('data/knnscore_data_0330/train/part_{}.csv'.format(i),'w') as writer:
            for line in reader:
                items = line.split(',')
                # print(','.join(items[1:]), file = writer)
                writer.write(','.join(items[1:]))
                # writer.write(','.join(items[1:]) + '\n') #这样写导致每隔一行都多一行空格


def dropFirstColumn1():
    with open('data/knnscore_data/test_all_100.csv','r') as reader, open('data/knnscore_data/test/test_joint_without_id_new.csv','w') as writer:
        for line in reader:
            items = line.split(',')
            # print(','.join(items[1:]), file = writer)
            writer.write(','.join(items[1:]))
            # writer.write(','.join(items[1:]) + '\n') #这样写导致每隔一行都多一行空格

def checkColumnNum():
    with open('data/knnscore_data/train/part_0.csv') as f:
        for line in f:
            line = line.split(',')
            print(len(line))


# def checkData():
#     data = []
#     for i in range(10):
#         filepath = "data/knnscore_data/train/part_{}".format(i)
#         with open(filepath, 'r') as fp:
#             data.append([x.split(',')[0] for x in fp])
# count = 0
# all= [] 
# for i in data:
#     print(len(i))
#     j = set(i)
#     all+=i
#     count += len(j)
#     print(len(j))
# print(len(set(all)))
# print(count)
#     return data


if __name__ == '__main__':
    # # load_data(CSV_FILE)
    # train, test = randomSplit(csv_file = CSV_FILE, pos_size=POS_SIZE, neg_size=NEG_SIZE, pick_rate=PICK_RATE)
    # protein_dataset = ProteinDataset(train, transform = ToTensor())
    # for i in range(len(protein_dataset)):
    #     properties, label = protein_dataset[i]
    #     print(i, type(properties), properties.size(), label)

    #     if i == 0:
    #         break;
    dropFirstColumn()
    # checkColumnNum()

