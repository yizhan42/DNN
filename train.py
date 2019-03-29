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

# from validate import *


train_loss_list = []
val_loss_list = []

train_accuracy_list = []
val_accuracy_list = []

def train(model,training, validation, args, times=0):
    total_accuracy = 0

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

    global sum_train
    # global sum_val
    torch.manual_seed(1)    # reproducible

    # put model in GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    # print("+++")
    # train_data = ProteinDataSet(
    train_data = ProteinDataset(
        pd_dataFrame = training,
        transform = ToTensor()
    )
   

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = torch.nn.NLLLoss(reduction='sum')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr)


# dynamic learning scheme
    if args.dynamic_lr and args.optimizer != 'Adam':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.decay_factor, last_epoch=-1)

    for epoch in range(EPOCH):
        loss = 0
        train_accuracy = 0
        size = 0
        for batch, (x, y) in enumerate(train_loader):# each batch
            size += len(y)
            # put data in GPU
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            b_x = Variable(x)  # batch x
            b_y = Variable(y.long(), requires_grad=False)  # batch y            
            y_score = model(b_x)
           
            train_loss = criterion(y_score, b_y)
            
            train_accuracy += sum(torch.max(y_score, 1)[1].data.squeeze() == b_y.data.long()).data.item()

            loss+=train_loss.data.item()
            
      # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            train_loss.backward()           
            optimizer.step()
           
        loss /=  size
        # print("train_accuracy:",train_accuracy)
        train_accuracy /= size
        
        # each epoch gets a val_accuracy and a validation_loss
        val_accuracy, validation_loss = validate(model, validation, args)

        train_loss_list.append(loss)
        val_loss_list.append(validation_loss)

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
       
        if args.log_result:
            with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
                r.write('{:10d} | {:10d} | {:10.7f} | {:10.7f} | {:10.7f} | {:10.7f} | {:10.8f}\n'.format(
                    epoch,
                    batch,
                    validation_loss,
                    loss,
                    val_accuracy,
                    train_accuracy,
                    optimizer.state_dict()['param_groups'][0]['lr']
                ))

        if (best_loss is None) or (validation_loss < best_loss):
            file_path = '{}/best_loss.pth.tar'.format(args.save_folder)
            print("=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'best_loss': best_loss,
                             'best_acc': best_acc},
                            file_path)

            best_loss = validation_loss

        if best_acc is None or val_accuracy > best_acc:
            file_path = '{}/best_accuracy.pth.tar'.format(args.save_folder)
            print("=> found better validated model, saving to %s" % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'best_loss': best_loss,
                             'best_acc': best_acc},
                            file_path)
            best_acc = val_accuracy

    # every EPOCH (100 epochs) get a average accuracy
    leng = len(val_accuracy_list)
    for i in range(EPOCH+1):
        if i == 0:
            pass
        total_accuracy += val_accuracy_list[-i]

    if epoch == EPOCH-1:
        print(leng)
        average_accuracy = total_accuracy / EPOCH
        lg('Average accuracy is : {:7.6f}'.format(average_accuracy))
        # total_accuracy = 0

    drawLossFigureFromFile(
        '{}/result.csv'.format(args.save_folder), is_print=False, is_save=True)



def save_checkpoint(model, state, filename):
    # model_is_cuda = next(model.parameters()).is_cuda
    # model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)


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
        
        train_dataset, validation_dataset = readTrainingData(
        label_data_path='{}/train/{}'.format(args.data_folder, args.prefix_filename),
        index=i,
        total=args.groups,
        # standard_length=args.length,
        )   
        # model = CNN_multihot()
        run_train(train_dataset, validation_dataset, model(), args)
    args.save_folder = save_folder
if __name__ == "__main__":
    args = parser.parse_args()
    run_main(CNN_multihot,args) # CNN_multihot是个类，CNN_multihot()是个对象，但是为了在十折交叉验证中每个折里面都新定义一个对象，所以此处用类，而在run函数中的每一折中定义一个对象


# run_main是 main 函数，run_main 中跑 run_train， run_train 中调用train函数