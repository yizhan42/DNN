# wdl模型只能处理 multihot 的数据，也就是4221维，算交叉内积时需要用到feature名字，所以跟函数给
# 所有数据加入第一行
def readcsv2list(file):
    rs = []
    with open(file,'r') as f:
        rs = f.readlines()
    return rs

def WDLdata():
    rs = readcsv2list()
    for i in range(4221):
        feature_list[i] = i



if __name__ == "__main__":
    WDLdata()