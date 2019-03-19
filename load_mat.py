# import scipy.io as scio

dataFile = 'E:\DeeplearningMissouri/Data_9_14.mat'
# data = scio.loadmat(dataFile)

import h5py
from h5py import Dataset,Group
import numpy as np

data = dict()
with h5py.File(dataFile, 'r') as f:
    print(list(f.keys()))# Datasets: ['#refs#', 'Data_All', 'NegaSetid']
    data = []
    # for items in f['#refs#']:
        # data.append(f['#refs#/'+items])
    # print(data)
        # if(type(f['#refs#/'+items])==Dataset):
        #     # continue
        #     print(f['#refs#/'+items])
        #     for i in f['#refs#/'+items]:
        #         print(i,)
        #     break
        # else:
        #     print(items)
            # for item in (f['#refs#/'+items]):

        # print(f['#refs#/'+items])
    for item in f['Data_All']:
        print(item[0])
        for i in f[item[0]]:
            print(i,type(i),f[item[0]][i])
            print(f[item[0]]['SQ'])
            sq = ''
            for key in f[item[0]]['SQ']:
                sq += (chr(key))
            print(sq)
            break
        break
        # for i in f[item[0]]:
        #     print(i)
    # for item in f['NegaSetid']:
    #     print(item.size, item.dtype)
    #     data = []
    #     for i in f[item[0]]:
    #         print(i)
    #         # data.append(i)
    #     print(data)
