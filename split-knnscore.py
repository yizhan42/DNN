import csv
import random


def readcsv2list(file):
    rs = []
    with open(file,'r') as f:
        rs = f.readlines()
    return rs

def writeFile(file1, file2, no, start, len, data, filter):
    print(start, start+len)
    # '{}_{}.csv'.format(file1, no)
    with open(file1+'_'+ str(no)+'.csv', 'a') as f1, open(file2+'_'+ str(no)+'.csv', 'a') as f2:
        for i, protein in enumerate(data):
            if i in filter[start:start+len]:
                f2.write(protein)
            else:
                f1.write(protein)

# 注意：只能执行一遍。把所有数据分为 train 和 test，
# train有 10710个，1-1710为正样本，1711-10710为负样本， 
# test 有 1196个, 1-196为正样本，197-1196为负样本
def train_test():
    data = readcsv2list('data/knnscore_data/final.csv')
    filter1 = [i for i in range(1906)]
    random.shuffle(filter1)
    filter2 = [i for i in range(10000)]
    random.shuffle(filter2)
    writeFile('data/knnscore_data/train_all', 'data/knnscore_data/test_all', 100, 0, 196, data[:1906], filter1) #pos
    writeFile('data/knnscore_data/train_all', 'data/knnscore_data/test_all', 100, 1906, 1000, data[1906:], filter2) #neg

def split10fold(data1, data2, file):
    with open(file, 'a') as f:
        for protein in data1:
            f.write(protein)
        for protein in data2:
            f.write(protein)

# train 和 test各自分成10份
def split():
    data1 = readcsv2list('data/knnscore_data/train_all_100.csv')
    data2 = readcsv2list('data/knnscore_data/test_all_100.csv')
    
    for i in range(10):
        # train有 10710个，[0,1710)1-1710为正样本，1711-10710为负样本， test有1196个, 1-196为正样本，197-1196为负样本
        if i < 9:           
            split10fold(data1[i * 171:i * 171 + 171], data1[1710 + i * 899:1711 + i * 899 + 899], 'data/knnscore_data/train/part_{}'.format(i))
            # split10fold(data2[i * 19:i * 19 + 19], data2[196 + i * 100:196 + i * 100 + 100], 'data/knnscore_data/test/part_{}'.format(i))
        else:
            split10fold(data1[i * 171:1710], data1[1710 + i * 899:10711], 'data/knnscore_data/train/part_{}'.format(i))
            # split10fold(data2[i * 19:197], data2[196 + i * 100:1197], 'data/knnscore_data/test/part_{}'.format(i))
    


if __name__ == "__main__":
    split()
    # train_test()

