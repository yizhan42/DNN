import argparse
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from scipy import interp
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from torch.autograd import Variable
from torch.nn import init

import analysis.figure
import torchvision
from analysis import *
from analysis import drawLossFigureFromFile
from analysis import write_and_print as wnp
from CNNnd import *
from CNNst import *
from load_data import *
from settings import *
from test_final import *
from validate import *

import errno

# from torch_model import *
from wide_deep.data_utils import prepare_data

# DF = pd.read_csv('data/wdl_data/train/train.csv', header=None)
wide_cols = [x for x in range(1,4221)]  
crossed_cols = ()
embeddings_cols = [(2,3),(3,4),(4,5)]
continuous_cols = [8, 9]
target = 0  
method = 'logistic'
hidden_layers = [100,50]
dropout = [0.5,0.2]

# wd_dataset = prepare_data(
#     DF, wide_cols,
#     crossed_cols,
#     embeddings_cols,
#     continuous_cols,
#     target,
#     scale=True)

def train(model, training, validation, args, times=0):
    model.compile(method=method, learning_rate = args.lr)
    # total_accuracy = 0

    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_loss = checkpoint.get('best_loss', None)
        best_acc = checkpoint.get('best_acc', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
        else:
            start_iter += 1
    else:
        best_loss = None
        best_acc = None

    torch.manual_seed(1)    # reproducible

    # put model in GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
   
    model.fit(args, training, validation, best_acc, best_loss, n_epochs=args.epochs, batch_size=64)

   
def run_train(train_dataset, validation_dataset, model, args):
    # Create save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory {} already exists.'.format(args.save_folder))
        else:
            raise

    # log result
    if args.log_result:
        result_fp = open(os.path.join(args.save_folder, 'result.csv'), 'a')
    else:
        result_fp = None
    # Display info of each sample

    wnp('Number of training samples: {}'.format(
        str(train_dataset.__len__())), result_fp, is_write=args.log_result)

    # log result title
    wnp('\n{:>10s} | {:>10s} | {:>10s} | {:>10s} | {:>10s} | {:>10s} | {:>10s}'.format(
        'epoch', 'batch', 'valid_loss', 'train_loss', 'valid_acc', 'train_acc', 'learn_rate'
    ), result_fp, is_print=False, is_write=args.log_result)

    if(args.log_result):
        result_fp.close()

    # train
    train(model, train_dataset, validation_dataset, args)



def run_main(model, args):
    # load training data
    print('Loading Configs...\n')
    if(args.start >= args.groups):
        print('Argument start should be smaller than groups. It is set to 0 forcedly.')
        args.start = 0
    if(args.end > args.groups or args.end <= args.start):
        print('Argument end should be no larger than groups and larger than start. It is set to start+1 forcedly.')
        args.end = args.start + 1
    # save the origial argument in case the values changed when running the model
    save_folder = args.save_folder
    val_loss_folder = args.val_loss_folder
    val_accuracy_folder = args.val_accuracy_folder
    args.cuda = torch.cuda.is_available() and args.cuda  # is cuda

    # Create save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory {} already exists.'.format(args.save_folder))
        else:
            raise

    # configuration
    log_fp = open('{}/arguments.md'.format(save_folder), 'w')
    wnp('Configuration:', log_fp)
    for attr, value in sorted(args.__dict__.items()):
        wnp('    {}:'.format(attr.capitalize().replace(
            '_', ' ')).ljust(25)+'{}'.format(value), log_fp)

    # load model
    # model = loadModule(args)
    wnp('\n{}'.format(model), log_fp)
    log_fp.close()

    # Run for each Cross Validation data
    for i in range(args.start, args.end):
        if args.start != args.end:
            args.save_folder = '{}/group_{}'.format(save_folder, i)
        # args.class_weight = class_weight
        print('Loading data from {}/train'.format(args.data_folder))
        train_df, validation_df = readTrainingData(label_data_path='{}/train/{}'.format(args.data_folder, args.prefix_filename), index=i, total=args.groups)
        #print(type(train_df[0][0]))
        train_dataset = prepare_data(
            train_df, wide_cols,
            crossed_cols,
            embeddings_cols,
            continuous_cols,
            target,
            scale=True)
        validation_dataset = prepare_data(
            validation_df, wide_cols,
            crossed_cols,
            embeddings_cols,
            continuous_cols,
            target,
            scale=True)
        wide_dim = train_dataset['dataset'].wide.shape[1]
        n_unique = len(np.unique(train_dataset['dataset'].labels))
        if (method=="regression") or (method=="logistic"):
            n_class = 1
        else:
            n_class = n_unique
        deep_column_idx = train_dataset['deep_column_idx']
        embeddings_input= train_dataset['embeddings_input']
        encoding_dict   = train_dataset['encoding_dict']
    
        
        run_train(train_dataset['dataset'], validation_dataset['dataset'], model(
            wide_dim,
            embeddings_input,
            continuous_cols,
            deep_column_idx,
            hidden_layers,
            dropout,
            encoding_dict,
            n_class
        ), args)
    args.save_folder = save_folder
if __name__ == "__main__":
    args = parser.parse_args()
    model = globals()[args.model] 
    run_main(model,args) # CNN_multihot是个类，CNN_multihot()是个对象，但是为了在十折交叉验证中每个折里面都新定义一个对象，所以此处用类，而在run函数中的每一折中定义一个对象


# run_main是 main 函数，run_main 中跑 run_train， run_train 中调用train函数
