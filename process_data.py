from __future__ import print_function, division

import csv
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils

dataFile = 'P_Ubi20180223xu.mat'

def run():
    import scipy.io as scio
    data = scio.loadmat(dataFile)
    for key in data:
        print(key)


    all = {'GO':[],'INTERPRO':[],
           'PFAM':[],'PRINTS':[],
           'PROSITE':[],'SMART':[],'SUPFAM':[]}

    documents = [data['P_Ubi_5fold'], data['P_Ubi_Duli'], data['P_UbiNeg']]
    for document in documents:
        for protein in document[0]:
            for feature in all:
                try:
                    for value in protein[feature][0,0][0]:
                        all[feature].append(value[0])#value属性编码

                except Exception as e:
                    # print(feature)
                    # print(e)
                    pass

    def write_csv(save_file, data_from , label):
        print(save_file)
        # below are title
        with open(save_file,'w') as fp:
            # aa = []
            all_values = ['id', 'label']
            # label = ['label']
            # # print(label)
            # all_values += label
            for feature in all:
                all_values += list(set(all[feature]))
            fp_csv = csv.writer(fp)
            fp_csv.writerow(all_values) #write first line
            # print(all_values)
            length = len(all_values)-2

            for protein in data[data_from][0]:
                # print(protein)
                p = np.zeros(length,dtype=float)
                ac = [protein['PRTID'][0,0][0], label]

                
                for feature in all:
                    try:
                        for value in protein[feature][0,0][0]:
                            i = all_values.index(value[0])
                            p[i] = float(1.0)
                    except Exception as e:
                        # print(feature)
                        # print(e)
                        pass
                
                fp_csv.writerow(ac +  p.tolist())

    write_csv('pos_train.csv','P_Ubi_5fold', 1)
    write_csv('neg.csv','P_UbiNeg', 0)
    write_csv('pos_test.csv', 'P_Ubi_Duli', 1)



CSV_FILE_POS = 'pos_train.csv'
SIZE_POS = 1600
PICK_RATE_POS = 0.2


def randomSplit_Pos(csv_file, size, pick_rate):
    val_list = random.sample(range(1, size + 1), int(pick_rate * size))
    i = 0

    trainf = open('pos_train_new.csv', 'w')
    valf = open('pos_val.csv', 'w')
    with open(csv_file, 'r') as fp:
        for line in fp:
            # line = line.replace('\r\n','').split(',')
            if (i == 0):
                trainf.write(line)
                valf.write(line)
            elif i in val_list:
                valf.write(line)
            else:
                trainf.write(line)

            i += 1

    trainf.close()
    valf.close()

CSV_FILE_NEG = 'less_neg.csv'
VAL_TEST_FILE = 'neg_val_test.csv'
# SIZE_NEG = 10000
# size = 3736
SIZE_NEG = 1906
size = 626
PICK_RATE_VT = 626/1906
PICK_RATE_T = 0.5

def lessNeg(csv_file):
    less = open('less_neg.csv','w')
    i = 0
    with open(csv_file) as fp:
        for line in fp:

            if i < 1907:
                less.write(line)
                i += 1

def randomSplit_Neg(csv_file , size, pick_rate):
    val_test_list = random.sample(range(1, size+1), int(pick_rate * size))
    i = 0

    trainf = open('neg_train.csv', 'w')
    val_testf = open('neg_val_test.csv', 'w')
    with open(csv_file, 'r') as fp:
        for line in fp:
            # line = line.replace('\r\n','').split(',')split
            if(i == 0):
                trainf.write(line)
                val_testf.write(line)
            elif i in val_test_list:
                val_testf.write(line)
            else:
                trainf.write(line)
            # if i in test_list:
            #     testf.write(line)
            # else:
            #     trainf.write(line)
            i += 1

    trainf.close()
    val_testf.close()



def randomSplit_val_test(csv_file, size, pick_rate):
    test_list = random.sample(range(1, size + 1), int(pick_rate * size))
    i = 0

    valf = open('neg_val.csv', 'w')
    testf = open('neg_test.csv', 'w')
    with open(csv_file, 'r') as fp:
        for line in fp:
            # line = line.replace('\r\n','').split(',')
            if (i == 0):
                valf.write(line)
                testf.write(line)
            elif i in test_list:
                testf.write(line)
            else:
                valf.write(line)

            i += 1

    valf.close()
    testf.close()


    

def merge_train(csv_file1,csv_file2 ):
    train = open('train_all.csv', 'w')
    i = 0

    with open(csv_file1, 'r') as fp:
        for line in fp:
            if i == 0:
                i+=1
                continue
            else:
                train.write(line)
            i += 0

    j = 0
    with open(csv_file2, 'r') as fq:

        for line in fq:
            if j == 0:
                j += 1
                continue
            else:
                train.write(line)
            j += 1

    train.close()

def merge_train_list():
    train_list = []
    with open('train_all.csv', 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            train_list.append(line)

    train_list = pd.DataFrame(data=train_list)
    return train_list


def merge_val(csv_file1, csv_file2):
    validation = open('val_all.csv', 'w')
    i = 0

    with open(csv_file1, 'r') as fp:
        for line in fp:
            if i == 0:
                i += 1
                continue
            else:
                validation.write(line)
            i += 1
    j = 0
    with open(csv_file2, 'r') as fq:
        for line in fq:
            if j == 0:
                j += 1
                continue
            else:
                validation.write(line)
            j += 1
    validation.close()

    

def merge_val_list():
    val_list = []
    with open('val_all.csv', 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            val_list.append(line)

    val_list = pd.DataFrame(data=val_list)
    return val_list

def merge_test(csv_file1, csv_file2):
    test = open('/home/chunhui/yi/DNN/NewData/data/test/test_all.csv', 'w')
    i = 0

    with open(csv_file1, 'r') as fp:
        for line in fp:
            if i == 0:
                i += 1
                continue
            else:
                test.write(line)
            i += 1
    j = 0
    with open(csv_file2, 'r') as fq:
        for line in fq:

            if j == 0:
                j += 1
                continue
            else:
                test.write(line)
            j += 1

    test.close()

   

def merge_test_list():
    test_list = []
    with open('/home/chunhui/yi/DNN/NewData/data/test/test_all.csv', 'r') as f:
        for line in f:
            line = line.replace('\r\n', '').split(',')
            test_list.append(line)

    test_list = pd.DataFrame(data=test_list)
    return test_list




def check():
    with open('pos_test.csv', 'r') as fp:
        count = 0
        for line in fp:
            if line == 0:
                print("title")
            elif len(line.split(','))  == 27340:
                print(count , line)
            count += 1




if __name__ == '__main__':

    run()
    # check()
    lessNeg(csv_file='neg.csv')
    randomSplit_Pos(csv_file=CSV_FILE_POS, size=SIZE_POS, pick_rate=PICK_RATE_POS)
    randomSplit_Neg(csv_file=CSV_FILE_NEG, size=SIZE_NEG, pick_rate=PICK_RATE_VT)
    randomSplit_val_test(csv_file=VAL_TEST_FILE, size=size, pick_rate=PICK_RATE_T)
    train = merge_train('pos_train.csv','neg_train.csv')
    validation = merge_val('pos_val.csv', 'neg_val.csv')
    test = merge_test('pos_test.csv', 'neg_test.csv')
    test_list = merge_test_list()