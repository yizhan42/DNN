# coding=utf8

import datetime

# Data Parameters
# TRAIN_CSV_FILE = '../data/train.csv'
# TEST_CSV_FILE = '../data/test.csv'
CSV_FILE = 'data/origin.csv'
NEG_SIZE = 422
POS_SIZE = 422
PICK_RATE = 0.1


#lalala
LOGGED = False
LOGGED = True
RUN_TIMES = 20
RUN_TIMES = 5
DATETIME = str(datetime.datetime.utcnow())
LOG_PATH = 'log/'
LOG_FILE = '{}log_{}.md'.format(LOG_PATH, DATETIME)
FIGURE_FILE = '{}figure_{}.png'.format(LOG_PATH, DATETIME)
# Hyper Parameters
EPOCH = 2000              # train the training data n times, to save time, we just train 1 epoch
# EPOCH = 1
BATCH_SIZE = 100
LR = 0.001              # learning rate

# IMAGE size
WIDTH = 14
HEIGHT = 14
# in_channels , out_channels, kernel_size, stride, padding, Max Pool
# input height, n_filters   , filter size, filter movement/step
CNN_P = [[1,16,5,1,2,2], [16,32,5,1,2,7]]
RS_Size = 32 * 1 * 1


# WIDTH = 175
# HEIGHT = 1
# # in_channels , out_channels, kernel_size, stride, padding, Max Pool
# # input height, n_filters   , filter size, filter movement/step
# CNN_P = [[1,16,(5,1),(1,1),(2,0),(7,1)], [16,32,(5,1),(1,1),(2,0),(25,1)]]
# RS_Size = 32*1*1#32 * 5 * 5

# WIDTH = 7
# HEIGHT = 25
# # in_channels , out_channels, kernel_size, stride, padding, Max Pool
# # input height, n_filters   , filter size, filter movement/step
# CNN_P = [[1,16,(5,5),(1,1),(2,2),(1,5)], [16,32,(5,5),(1,1),(2,2),(7,5)]]
# RS_Size = 32*1*1#32 * 5 * 5

# Hout=floor((Hin+2∗padding[0]−dilation[0]∗(kernel_size[0]−1)−1)/stride[0]+1)
# Wout=floor((Win+2∗padding[1]−dilation[1]∗(kernel_size[1]−1)−1)/stride[1]+1)
Class_N = 2

IS_WRITABLE = True
IS_PRINTABLE = True

if(LOGGED):
    log = open(LOG_FILE, 'w')
def lg(string, p = IS_PRINTABLE, w = IS_WRITABLE):
    if p:
        print(string)
    if LOGGED and w:
        log.write(string + "\n")

lg("Data scv file: {}\n".format(CSV_FILE))
