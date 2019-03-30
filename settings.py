# coding=utf8

import datetime
import argparse




LOGGED = False
LOGGED = True
RUN_TIMES = 1
DATETIME = str(datetime.datetime.utcnow())
LOG_PATH = 'log/'
LOSS_PATH = 'train_val_loss/'
ACCURACY_PATH = 'train_val_accuracy/'
LOG_FILE = '{}log_{}.md'.format(LOG_PATH, DATETIME)
FIGURE_FILE = '{}figure_{}.png'.format(LOG_PATH, DATETIME)
# Hyper Parameters
EPOCH = 100           # train the training data n times, to save time, we just train 1 epoch
# EPOCH = 1
BATCH_SIZE = 128
LR = 0.001              # learning rate


Class_N = 2

IS_WRITABLE = True
IS_PRINTABLE = True
IS_SAVE = True
IS_PRINT = True

parser = argparse.ArgumentParser(description='Predict Ubiquitination with DNN')
# parser.add_argument('--model-path', default='result', help='Path to pre-trained acouctics model created by DeepSpeech training')
parser.add_argument('--model-path', default='best_param/best_accuracy', help='Path to pre-trained acouctics model created by DeepSpeech training')

# Logging options
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

# data.add_argument('--cross-validation', default=False, action='store_true', help='Apply cross validation to reduce the random error.')
data = parser.add_argument_group('Data options')
data.add_argument('--groups', default=10, type=int,
    help='The number of the training groups.')
data.add_argument('--start', default=0, type=int,
    help='Start gourp of this running. It must be smaller than groups.')
data.add_argument('--end', default=10, type=int,
    help='End group of this running. Not including this group. It must not be largers than groups.')


# common learning options
learn = parser.add_argument_group('Learning options')
learn.add_argument('--model', default='CNN1',
                   help='Type of model. {} are supported [default: CNN1]'.format(models))
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



# cross validation data path
# data.add_argument('--train-data-folder', default='/home/chunhui/yi/DNN/NewData/data/train/', metavar='DIR',
#     help='path of training data folder')
data.add_argument('--train-data-folder', default='./data/train/', metavar='DIR',
    help='path of training data folder')
# data.add_argument('--test-data-folder', default='/home/chunhui/yi/DNN/NewData/data/test/', metavar='DIR',
#     help='path of training data folder')
data.add_argument('--test-data-folder', default='./data/test/', metavar='DIR',
    help='path of training data folder')
data.add_argument('--prefix-filename', default='part', metavar='DIR',
    help='prefix filename of splited cross validation data csv. If it\'s "part", then filename is "part_{}.csv".format(index).')
data.add_argument('--length', default=300, type=int,
    help='The standard length of each sequence')

# device
device = parser.add_argument_group('Device options')
device.add_argument('--num-workers', default=0, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', default=False, action='store_true', help='enable the gpu' )


if(LOGGED):
    log = open(LOG_FILE, 'w')
def lg(string, p = IS_PRINTABLE, w = IS_WRITABLE):
    if p:
        print(string)
    if LOGGED and w:
        log.write(string + "\n")

# lg("Data scv file: {}\n".format(CSV_FILE))
