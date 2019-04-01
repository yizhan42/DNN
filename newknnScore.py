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
#from settings import *
from numpy import genfromtxt
def combine():
    features = ['SSF', 'SM', 'PS', "PR", "PF", "IPR", 'GO']
    ffp = [open('data/distance_balanced/ID.csv', 'r')] + [open('data/distance_balanced/{}_score.csv'.format(feature), 'r') for feature in features[::-1]]
    with open('data/distance_balanced/final.csv', 'w') as wp:
        for i in range(3812):
            line = ''
            if i % 100 :
                print(i)
            for fp in ffp:
                line += fp.readline()[:-1]+','
            wp.write(line[:-1]+'\n')

def filterKnnScore():
    np.set_printoptions(precision=4)
    features = ['SSF', 'SM', 'PS', "PR", "PF", "IPR", 'GO']
    ids = genfromtxt('data/distance_balanced/ID.csv', delimiter=',')
    print("load id down")
    for feature in features:
        X = genfromtxt('data/distance_balanced/{}_rank.csv'.format(feature), dtype=int, delimiter=',')
        total_number = len(X)
        print('load {} {} data down'.format(total_number, feature))
        
        with open('data/distance_balanced/{}_score.csv'.format(feature),'w') as f:
            writer = csv.writer(f, delimiter=',')
            for i in range(total_number):
                protein_id = ids[i][0]
                label = ids[i][1]
                end = int(total_number * 29 / 100) + 1
                gap = int(total_number/100)
                threshold = max(5*gap, 1)
                gap = max(gap, 1)
                same_label = 0
                result = []
                for idx in range(end):
#                    print(X[i], X[i][idx])
#                    print(ids)
                    another_protein_id = ids[X[i][idx]][0]
                #   print(idx, another_protein_id, df[1][indices[idx]])
                    if label == ids[X[i][idx]][1]:
                        same_label += 1
                #   print(type(idx), type(threshold), idx==threshold)
                    if idx+1 == threshold:
                        result.append(same_label/(idx+1))
                        threshold += gap
                #print(result)
                writer.writerow(result)


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

def splitDataByFeature(FILE):
    fea_breakpoint = {'ID':[0,2], 'GO': [2, 10068], 'IPR': [10068, 19110], 'PF': [19110, 23196], 'PR': [23196, 23897], 'PS': [23897, 25579], 'SM': [25579, 26374], 'SSF': [26374, 27340]}        
    with open(FILE, 'r') as fp:
        for feature in fea_breakpoint.keys():
            fea_breakpoint[feature].append(csv.writer(open("data/distance_balanced/{}.csv".format(feature), 'w')))
        for line in fp:
            line = line[:-1].split(',')
            for start,end,wp in fea_breakpoint.values():
                wp.writerow(line[start:end])
def sort_protein(ids, feature):
    np.set_printoptions(precision=4)
    X = np.asmatrix(genfromtxt('data/distance_balanced/{}.csv'.format(feature), delimiter=','))
    with open('data/distance_balanced/{}_rank.csv'.format(feature),'w') as f, open("data/distance_balanced/{}_distance.csv".format(feature), 'w') as df :
        writer = csv.writer(f)
        dfwriter = csv.writer(df)
        protein_len = len(X)
        print(feature, protein_len)
        intersection = X*X.T
        print('intersection down')
        cot = X.sum(1)
        print('cot down')
        union = cot+cot.T-intersection
        print('union down')
        result = np.where(union==0, 1, 1-intersection/union)
        dfwriter.writerows(result)
        print('result down')
        writer.writerows(np.argsort(result))
        #writer.writerows(np.sort(result))
        print('Sort down\n')
'''
def sort_protein(ids, feature):
    np.set_printoptions(precision=4)
    X = np.asmatrix(genfromtxt('data/feature/{}.csv'.format(feature), delimiter=','))
    with open('data/distances/{}_distance.csv'.format(feature),'w') as f:
        writer = csv.writer(f)
        protein_len = len(X)
        print(feature, protein_len)
        intersection = X*X.T
        print('intersection down')
        cot = X.sum(1)
        print('cot down')
        union = cot+cot.T-intersection
        print('union down')
        result = np.where(union==0, 0.0001, 1-intersection/union)
        print('result down')
        writer.writerows(result)
'''
# 求距离矩阵
def main():        
    fea_breakpoint = {'GO': [2, 10068], 'IPR': [10068, 19110], 'PF': [19110, 23196], 'PR': [23196, 23897], 'PS': [23897, 25579], 'SM': [25579, 26374], 'SSF': [26374, 27340]}
    fp = open('data/distance_balanced/ID.csv', 'r')
    ids = [line[:-1].split(',') for line in fp]
    features = ['SSF', 'SM', 'PS', "PR", "PF", "IPR", 'GO']
    protein_len = len(ids)
    print("protein_len", protein_len)
    for feature in features:
        print("feature:",feature)
        sort_protein(ids, feature)
        print("write down")

# 求KNN-score矩阵，返回训练样本。先把数据存为dataFrame格式，然后pd.to_csv()将训练样本保存为csv格式
'''
def filterKnnScore(i, indices, df, writer):
    total_number = len(df)
    protein_id = df[0][i]
    end = max(int(total_number * 29 / 100) + 1, len(indices))
    gap = int(total_number/100)
    threshold = max(5*gap, 1)
    gap = max(gap, 1)
    same_label = 0
    result = [protein_id, df[1][i]]
    #print("from 1 to" ,end, "start at", threshold ,"step by", gap)
    for idx in range(1,end):
        another_protein_id = df[0][indices[idx]]
     #   print(idx, another_protein_id, df[1][indices[idx]])
        if result[1] == df[1][indices[idx]]:
            same_label += 1
     #   print(type(idx), type(threshold), idx==threshold)
        if idx == threshold:
            result.append(same_label/idx)
            threshold += gap
    #print(result)
    writer.writerow(result)
'''
def write_knnScore_tocsv():
    with open('data/knnScore_data.csv','w') as f:
        writer = csv.writer(f)
        for value in fea_result.values():
            writer.writerow(value)

if __name__ == '__main__':
    # getData()
    # getDist('origin_all_data.csv')
    # Y = getDist('short_pos_test.csv')
    # getKnnScore([[1,3,5,7,9]],Y)
    # getBreakPoint()
    # getLineNum('test.csv')
    #main('short_pos_test.csv')
    #main('nofirstline_postest.csv')
    #main()
#    splitDataByFeature('data/distance_balanced/origin_balanced_data.csv')
    main()
#    filterKnnScore()
#    combine()


