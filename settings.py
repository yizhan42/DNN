# coding=utf8
# author=yu

import datetime

# Data Parameters
# TRAIN_CSV_FILE = '../data/train.csv'
# TEST_CSV_FILE = '../data/test.csv'
CSV_FILE = 'data/origin.csv'
NEG_SIZE = 500
POS_SIZE = 422
PICK_RATE = 0.1

LOGGED = False
LOGGED = True
RUN_TIMES = 10
DATETIME = str(datetime.datetime.utcnow())
LOG_PATH = 'log/'
LOG_FILE = '{}log_{}.md'.format(LOG_PATH, DATETIME)
FIGURE_FILE = '{}figure_{}.png'.format(LOG_PATH, DATETIME)
# Hyper Parameters
EPOCH = 1000             # train the training data n times, to save time, we just train 1 epoch
# EPOCH = 1
BATCH_SIZE = 50
LR = 0.001              # learning rate

# IMAGE size
WIDTH = 1
HEIGHT = 175
# in_channels , out_channels, kernel_size, stride, padding, Max Pool
# input height, n_filters   , filter size, filter movement/step
CNN_P = [[1,16,5,1,2,4], [16,32,5,1,2,4]]
RS_Size = 32 * 1 * 1
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
