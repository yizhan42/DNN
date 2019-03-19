#coding:UTF-8
from __future__ import print_function, division
import scipy.io as scio
import numpy as np
import csv
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.cuda
from sklearn.neighbors import NearestNeighbors
from settings import *

dataFile = './P_Ubi20180223xu.mat'


def getData():
    data = scio.loadmat(dataFile)
    
    all = {'GO':[],'INTERPRO':[],
           'PFAM':[],'PRINTS':[],
           'PROSITE':[],'SMART':[],'SUPFAM':[]}

    documents = [data['P_Ubi_5fold'], data['P_Ubi_Duli'], data['P_UbiNeg']]
    for document in documents:
        for protein in document[0]:
            for feature in all:
                try:
                    for value in protein[feature][0,0][0]:
                        all[feature].append(value[0]) # value属性编码
                except Exception as e:
                    pass

    def write_csv(save_file, data_from , label):
        print(save_file)
        # below are title
        with open(save_file,'w') as fp:
            all_values = ['id', 'label']
         
            for feature in all:
                all_values += list(set(all[feature]))
            fp_csv = csv.writer(fp)
            fp_csv.writerow(all_values) #write first line
         
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
                        pass
                
                fp_csv.writerow(ac +  p.tolist())
    
    write_csv('pos_train.csv','P_Ubi_5fold', 1)
    write_csv('neg.csv','P_UbiNeg', 0)
    write_csv('pos_test.csv', 'P_Ubi_Duli', 1)
    
index_list = []
dist_list = []
fea_result = dict() 
# fea_label = {}


# 求feature断点
def getBreakPoint():
    with open('pos_test.csv','r') as csvfile:
        reader = csv.reader(csvfile)
        for i,rows in enumerate(reader):
            if i == 0:
                row = rows
    # print(row)
    fea_breakpoint = {'GO':[2,0], 'IPR':[0,0], 'PF':[0,0], 'PR':[0,0], 'PS':[0,0], 'SM':[0,0], 'SSF':[0,0]}
    # GO = [], IPR = [], PF = [], PR = [], PS = [], SM = [], SSF = []
    # count_go = 0, count_ipr = 0, count_pf = 0, count_pr = 0, count_ps = 0, count_sm = 0,count_ssf = 0

    for i in range(2, len(row)):
        for feature in fea_breakpoint.keys():
            if row[i].startswith(feature):
                if row[i-1].startswith(feature):
                    fea_breakpoint[feature][1] = i + 1
                else:
                    fea_breakpoint[feature][0] = i
    fea_breakpoint['GO'][1] = fea_breakpoint['IPR'][0]
    print(fea_breakpoint)
    return fea_breakpoint
        






# 求距离矩阵
def getDist(FILE):
    df = pd.read_csv(FILE, sep=',')
    Y = np.array(df.values[:,0:2]) # Y means ID and label
    for i in range(len(Y)):
        fea_result[Y[i][0]] = [Y[i][0], Y[i][1]] # in dict fea_label, key means ID, value means [id, label, score ...]
        

    fea_breakpoint = {'GO': [2, 10068], 'IPR': [10068, 19110], 'PF': [19110, 23196], 'PR': [23196, 23897], 'PS': [23897, 25579], 'SM': [25579, 26374], 'SSF': [26374, 27340]}
    for feature, break_point in fea_breakpoint.items():
        X = np.array(df.values[:,break_point[0]:break_point[1]])
        print(len(X))
        # if torch.cuda.is_available():
        #     print('cuda is available \n')
        #     X = X.cuda()

        nbrs = NearestNeighbors(n_neighbors=29, algorithm='ball_tree').fit(X) # 29 means threshold
        distances, indices_list = nbrs.kneighbors(X)
        getKnnScore(indices_list, total_number=11906)
        write_knnScore_tocsv()

    # # print(fea_label)
    # return distances, indices

# 求KNN-score矩阵，返回训练样本。先把数据存为dataFrame格式，然后pd.to_csv()将训练样本保存为csv格式

def getKnnScore(indices_list, total_number = 11906):
    # indices_list 是某个feature 下所有的nn index
    # knnScores = {} # protein id - knn score list
    for indices in indices_list:
        protein_id = indices[0]
        # knnScores[protein_id] = []
        end = int(total_number * 29 / 100) + 1
        gap = total_number/100
        threshold = 5*gap
        # thresholds = [int(i * total_number/100) for i in range(5,30)]
        # threshold_idx = 0
        same_label = 0
        for idx in range(1,end):
            another_proteind_id = indices[idx]
            if fea_result[protein_id][1] == fea_result[another_proteind_id][1]:
                same_label += 1
            if idx == threshold:
                fea_result[protein_id].append(same_label/idx)
                threshold += gap
    # return knnScores
            
    # for i in range(len(fea_label)):
    # for key, value in fea_label.items(): # all samples
    #     label = value
    #     for i in range(30): # all thresholds
    #         same_label = 0
    #         if i < 5:
    #             continue
    #         else:
    #             threshold_list.append(indices[0:i+1])
    #             temp_list = index_list[0][0:i+1]
    #             print(temp_list)
    #             print("***")
    #             print(temp_list[0])
    #             same_label = 0
    #             for j in range(len(temp_list[0])):
    #             # for j in range(len(indices[0])):

    #                 # if fea_label[temp_list[0][j]] == value:
    #                 if temp_list[0][j] == key:
    #                 # if threshold_list[0][j] == key:
    #                     same_label = same_label+1
    #                 j = j + 1
    #             knn_score = same_label / len(index_list[0])
    #             # knn_score = same_label / len(indices)
    #             knnScore_dict[key] = knn_score
    # print(knnScore_dict)
    # return knnScore_dict


def write_knnScore_tocsv():
    with open('knnScore_data.csv','w') as f:
        writer = csv.writer(f)
        for value in fea_result.values():
            writer.writerow(value)


if __name__ == '__main__':
    # getData()
    # getDist('origin_all_data.csv')
    getDist(CSV_FILE)
    # getKnnScore()
    # getBreakPoint()