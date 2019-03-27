from __future__ import print_function, division

import h5py
import numpy as np
import csv
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
from FeatureSelectionData import *

dataFile = 'P_Ubi20180223xu.mat'

def statistics():
    with h5py.File(dataFile, 'r') as f:
        char_list = list()
        for item in f['Data_All']:
            for key in f[item[0]]['SQ']:
                if chr(key) not in char_list:
                    char_list.append(chr(key))
        print(char_list)

def readMatWriteCsv():
    with h5py.File(dataFile, 'r') as f:
        print(list(f.keys()))# Datasets: ['#refs#', 'Data_All', 'NegaSetid']

        print(f['Data_All'].size)
        with open('sequence.csv', 'w') as cf:
            writer = csv.writer(cf, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # writer.writerow(['id','label','sequence'])
            char_list = list()
            for item in f['Data_All']:
                sq = ''
                id = ''
                label = str(int(f[item[0]]['Label'][0][0]))
                for key in f[item[0]]['SQ']:
                    sq += (chr(key))
                    if chr(key) not in char_list:
                        char_list.append(key)
                # print(sq)
                for key in f[item[0]]['ID']:
                    id += (chr(key))
                writer.writerow([id,label,sq])
                print(char_list)
                

class ProteinDataSet(Dataset):
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

    def __call__(self, sample):
        properties, label = sample

        # convert np.ndarray to tensor
        properties = torch.from_numpy(properties)
        # insert depth
        properties = properties.float()#.view(1, WIDTH, HEIGHT)

        return properties, label

# def readTrainingData(label_data_path='/home/chunhui/yi/DNN/NewData/data/train/part', index=0, total=10):
def readTrainingData(label_data_path='./data/train/part', index=0, total=10):
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
def readTestData(label_data_path='./data/test/part', index=0, total=10):
   
    test_data = []
    for i in range(total):
        with open('{}_{}.csv'.format(label_data_path, i)) as read_file:
            reader = csv.reader(read_file)
            if i == index:
                for item in reader:
                    test_data.append(item)

    test_data = pd.DataFrame(data = test_data)
    # test_small_data = pd.DataFrame(data = test_small_data)
    return test_data


        
#     P_Ubi
# P_Ubi_Duli
# __globals__
# P_Ubi_5fold
# P_UbiNeg
# ans
# __header__
# __version__
    # readMatWriteCsv()
    # statistics()
if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset, validation_dataset = readTrainingData(
        label_data_path='{}{}'.format(args.train_data_folder, args.prefix_filename),
        index=0,
        total=args.groups,
        # standard_length=args.length,
    )
    train_data = ProteinDataSet(
        pd_dataFrame=train_dataset,
        transform=ToTensor()
    )
    model = TwoLayerNet(D_in, H, I, J, K, D_out)
    train(model, train_dataset, validation_dataset, args)
    # readTrainingData()