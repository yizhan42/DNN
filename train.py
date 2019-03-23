import argparse
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve

import analysis.figure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from analysis import *
from CNNnd import *
from CNNst import *
# from FeatureSelectionData import *
from load_data import *
# from model import *
# from process_data import *
from settings import *
from test_final import *
from validate import *
from torch.autograd import Variable
from torch.nn import init
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

    print("+++")
    # train_data = ProteinDataSet(
    train_data = ProteinDataset(
        pd_dataFrame = training,
        transform = ToTensor()
    )
   

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = torch.nn.NLLLoss()

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
        size = 0
        for step, (x, y) in enumerate(train_loader):# each batch
            size += len(y)
            # put data in GPU
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            b_x = Variable(x)  # batch x
            b_y = Variable(y.long(), requires_grad=False)  # batch y            
            y_score = model(b_x)
            # y_score = Variable(y_score)
            # print("y_score:",y_score)
            train_loss = criterion(y_score, b_y)
            # y_pred = torch.max(y_score, 1)[0]
            # print("y_pred:",y_pred)
            
            train_accuracy = sum(torch.max(y_score, 1)[1].data.squeeze() == b_y.data.long()) / float(
                b_y.size(0))

            
            # loss+=train_loss.data[0]
            loss+=train_loss.data.item()
            
      # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            train_loss.backward()           
            optimizer.step()
           
        loss /=  size
        
        # each epoch gets a val_accuracy and a validation_loss
        val_accuracy, validation_loss = validate(model, validation, args)

        train_loss_list.append(loss)
        val_loss_list.append(validation_loss)

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)

        
        if args.log_result:
            with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
                r.write('train_loss:{:10.7f} | validation_loss:{:10.7f} | val_accuracy:{:3.2f}\n'.format(
                    loss,
                    validation_loss,
                    val_accuracy
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

       
        if epoch % 10 == 0:
            lg('    Epoch: {:4d} | train loss: {:7.6f}  | validation loss: {:7.6f} | validation accuracy: {:3.2f}'.format(
                epoch, loss , validation_loss, val_accuracy))
        
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

    
    # print("\n {}".format(print_f_score(y_pred, b_y, average="weighted")))
    # print("\n {}".format(accuracy_score(model.predict(test_dataset), test_dataset.labels)))


def bundle(model, times = 10):
    args = parser.parse_args()
     
    train_dataset, validation_dataset = readTrainingData(
        label_data_path='{}{}'.format(args.train_data_folder, args.prefix_filename),
        index=i,
        total=args.groups,
        # standard_length=args.length,
    )   
    train(model,train_dataset,validation_dataset,args)   
    plt_loss(args, train_loss_list, val_loss_list)

def save_checkpoint(model, state, filename):
    # model_is_cuda = next(model.parameters()).is_cuda
    # model = model.module if model_is_cuda else model
    state['state_dict'] = model.state_dict()
    torch.save(state,filename)

if __name__ == "__main__":
    print(BATCH_SIZE)
    i = 0
    for i in range(RUN_TIMES) :
        bundle(CNN_multihot)
