import random
testNo = [4, 17, 20, 37, 42, 48, 51, 56, 63, 67, 77, 91, 94, 98, 103, 106, 110, 120, 162, 163, 177, 183, 196, 203, 204, 205, 207, 210, 220, 239, 243, 244, 250, 284, 299, 310, 317, 335, 350, 352, 366, 390, 407, 410, 411, 414, 418, 421, 454, 475, 509, 515, 520, 553, 557, 559, 560, 564, 565, 590, 601, 625, 633, 647, 654, 655, 661, 664, 671, 685, 698, 737, 743, 752, 754, 764, 781, 803, 807, 828, 855, 860, 864, 878, 893, 895, 901, 904, 908, 909, 913, 916]
# Pick random data
# testNo = []
# i = 0
# while(i<50):
#     ra = random.randint(1,500)
#     if ra not in testNo:
#         testNo.append(ra)
#         i += 1
# i = 0
# while(i<42):
#     ra = random.randint(501,922)
#     if ra not in testNo:
#         testNo.append(ra)
#         i += 1
# testNo.sort()
# print(testNo,len(testNo))

i = 1
train_file = open('train.csv', 'w')
test_file = open('test.csv', 'w')
with open("testdata.csv",'r') as fp:
    for line in fp:
        test_file.write(line)if i in testNo else train_file.write(line)
        i += 1
