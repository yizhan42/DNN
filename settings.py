# coding=utf8

import argparse
import datetime

# Data Parameters
# TRAIN_CSV_FILE = '../data/train.csv'
# TEST_CSV_FILE = '../data/test.csv'
# CSV_FILE = 'data/origin.csv'
CSV_FILE = 'origin_all_data.csv'
# NEG_SIZE = 422
# POS_SIZE = 422
# PICK_RATE = 0.1

LOGGED = False
LOGGED = True
RUN_TIMES = 20
RUN_TIMES = 5
DATETIME = str(datetime.datetime.utcnow())
LOG_PATH = 'log/'
LOG_FILE = '{}log_{}.md'.format(LOG_PATH, DATETIME)
FIGURE_FILE = '{}figure_{}.png'.format(LOG_PATH, DATETIME)
# Hyper Parameters
EPOCH = 200             # train the training data n times, to save time, we just train 1 epoch
# EPOCH = 1
BATCH_SIZE = 100
LR = 0.001              # learning rate

# IMAGE size
WIDTH = 14
HEIGHT = 14
# in_channels , out_channels, kernel_size, stride, padding, Max Pool
# input height, n_filters   , filter size, filter movement/step
CNN_P = [[1,16,5,1,2,2], [16,32,5,1,2,7]]
RS_Size = 9632 * 1 * 1


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



parser = argparse.ArgumentParser(description='Predict Ubiquitination with CNN')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num-workers', default=0, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', default=False, action='store_true', help='enable the gpu' )

# cross validation data path
data = parser.add_argument_group('Data options')
data.add_argument('--data-folder', default='data/multihot_data/', 
    help='path of training data folder')
data.add_argument('--cross-validation', default=False, action='store_true', help='Apply cross validation to reduce the random error.')

data.add_argument('--groups', default=10, type=int,
    help='The number of the training groups.')
data.add_argument('--start', default=0, type=int,
    help='Start gourp of this running. It must be smaller than groups.')
data.add_argument('--end', default=10, type=int,
    help='End group of this running. Not including this group. It must not be largers than groups.')
data.add_argument('--train-data-folder', default='./data/multihot_data/', 
    help='path of training data folder')
# data.add_argument('--test-data-folder', default='/home/chunhui/yi/DNN/NewData/data/test/', metavar='DIR',
#     help='path of training data folder')
data.add_argument('--test-data-folder', default='./data/multihot_data/test/test_joint_without_id.csv',
    help='path of training data folder')
data.add_argument('--prefix-filename', default='part', 
    help='prefix filename of splited cross validation data csv. If it\'s "part", then filename is "part_{}.csv".format(index).')
data.add_argument('--length', default=175, type=int,
    help='The length of data column number')

 #Logging options
experiment = parser.add_argument_group('Logging options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False, help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--continue-from', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true', help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint-per-batch', default=10000, type=int, help='Save checkpoint per batch. 0 means never save [default: 10000]')
experiment.add_argument('--save-folder', default='best_param', help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--val-loss-folder', default='train_val_loss', help='Location to save each cross validation loss.')
experiment.add_argument('---val-accuracy-folder', default='train_val_accuracy', help='Location to save each cross validation accuracy.')
experiment.add_argument('--log-config', default=True, action='store_false', help='Store experiment configuration')
experiment.add_argument('--log-result', default=True, action='store_false', help='Store experiment result')
experiment.add_argument('--log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--val-interval', type=int, default=200, help='how many steps to wait before vaidation [default: 200]')
experiment.add_argument('--save-interval', type=int, default=1, help='how many epochs to wait before saving [default:1]')
experiment.add_argument('--print-interval', type=int, default=1, help='how many epochs to wait before printing status [default:1]')


# common learning options
learn = parser.add_argument_group('Learning options')
learn.add_argument('--model', default='DNN_knnscore',
                   help='Type of model. {} are supported [default: DNN]')
learn.add_argument('--num-class', type=int, default=2,help='The number of classes.')


learn.add_argument('--epochs', type=int, default=200,
                   help='number of epochs for train [default: 200]')
learn.add_argument('--train-batch-size', type=int, default=128,
                   help='Batch size for training [default: 128]')
learn.add_argument('--validation-batch-size', type=int,
                   default=128, help='batch size for validation [default: 128]')
# learn.add_argument('--max-norm', default=400, type=int,
#                    help='Norm cutoff to prevent explosion of gradients')

learn.add_argument('--optimizer', default='Adam',
                   help='Type of optimizer. SGD|Adam|ASGD are supported [default: Adam]')


# learning rate
learn.add_argument('--lr', type=float, default=0.0010,
                   help='initial learning rate [default: 0.001]')
learn.add_argument('--dynamic-lr', default=False,
                   action='store_true', help='Use dynamic learning schedule.')
learn.add_argument('--milestones', default=[100, 200, 250], nargs='+', type=int,
                   help=' List of epoch indices. Must be increasing. Default:[5,10,15]')
learn.add_argument('--decay-factor', default=0.1, type=float,
                   help='Decay factor for reducing learning rate [default: 0.5]')

parser.add_argument('--model-path', default='best_param')
    
parser.add_argument('--prefix-groupname', default='group', metavar='DIR',
    help='prefix name of each tained model. If it\'s "group", then path is "{}/group_{}/best.pth.tar".format(model_path, index).')
log = open(LOG_FILE, 'w')
def lg(string, p = IS_PRINTABLE, w = IS_WRITABLE):
    if p:
        print(string)
    if LOGGED and w:
        log.write(string + "\n")

# lg("Data csv file: {}\n".format(CSV_FILE))
