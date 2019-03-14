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
from settings import *

# Load Data with testing and training data randomly splited.
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
        # extract different part of data - name, label and properties
        self.labels = self.properties_frame.ix[:,1].values().astype('int64')
        self.labels[self.labels==-1] = 0 # modify -1 to 0
        self.protein_names = self.properties_frame.ix[:,0]
        self.properties_frame = self.properties_frame.ix[:,2:]
        self.transform = transform

    def __len__(self):
        return len(self.properties_frame)

    def __getitem__(self, idx):
        # convert pd.DataFrame to np.ndarray or other
        protein_name = self.protein_names.ix[idx,:]
        properties = self.properties_frame.ix[idx,:].values().astype('double')
        label = self.labels[idx]

        properties.resize((WIDTH, HEIGHT))
        sample = (properties, label)

        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        properties, label = sample

        # convert np.ndarray to tensor
        properties = torch.from_numpy(properties)
        # insert depth
        properties = properties.float().view(1, WIDTH, HEIGHT)
        # properties = properties.float().view(1, HEIGHT, WIDTH)
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

if __name__ == '__main__':
    # # load_data(CSV_FILE)
    train, test = randomSplit(csv_file = CSV_FILE, pos_size=POS_SIZE, neg_size=NEG_SIZE, pick_rate=PICK_RATE)
    protein_dataset = ProteinDataset(train, transform = ToTensor())
    for i in range(len(protein_dataset)):
        properties, label = protein_dataset[i]
        print(i, type(properties), properties.size(), label)

        if i == 0:
            break;
