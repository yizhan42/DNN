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
from FeatureSelectionData import *
from load_data import *
from model import *
from process_data import *
from settings import *
from test_final import *
from torch.autograd import Variable
from torch.nn import init
from validate import *


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H,I,J,K, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()

    #self.input = torch.nn.Dropout(p = 0.75)
    self.linear1 = torch.nn.Linear(D_in, H)
    # torch.manual_seed(1)
    # init.xavier_normal(self.linear1.weight)
    # print(self.linear1.weight.data)
    # std = math.sqrt(2) / math.sqrt(7.)
    # self.linear1.weight.data.normal_(0, std)


    #self.linear1_bn = torch.nn.BatchNorm1d(H)
    #self.dropout = nn.Dropout(p=0.5)
    #self.H = nn.ReLU()

    # m = nn.ReLU()
    # input = autograd.Variable(torch.randn(2))
    # print(m(input))


    #H = Variable(H)
    self.linear2 = torch.nn.Linear(H, I)
    # torch.manual_seed(1)
    # init.xavier_normal(self.linear2.weight)
    # print(self.linear2.weight.data)
    # std = math.sqrt(2) / math.sqrt(7.)
    # self.linear2.weight.data.normal_(0, std)

    self.linear3 = torch.nn.Linear(I, J)
    # torch.manual_seed(1)
    # init.xavier_normal(self.linear3.weight)
    # print(self.linear3.weight.data)
    # std = math.sqrt(2) / math.sqrt(7.)
    # self.linear3.weight.data.normal_(0, std)

    self.linear4 = torch.nn.Linear(J, K)
    # torch.manual_seed(1)
    # init.xavier_normal(self.linear4.weight)
    # print(self.linear4.weight.data)
    # std = math.sqrt(2) / math.sqrt(7.)
    # self.linear4.weight.data.normal_(0, std)

    self.linear5 = torch.nn.Linear(K, D_out)
    # torch.manual_seed(1)
    # init.xavier_normal(self.linear5.weight)
    # print(self.linear5.weight.data)
    # std = math.sqrt(2) / math.sqrt(7.)
    # self.linear5.weight.data.normal_(0, std)

    # from torch.nn import init
    #
    # linear = nn.Linear(3, 4)
    #
    # t.manual_seed(1)
    #
    # init.xavier_normal(linear.weight)
    # print(linear.weight.data)
    #
    # import math
    #
    # std = math.sqrt(2) / math.sqrt(7.)
    # linear.weight.data.normal_(0, std)

    # try new torch.nn.functional.linear(input, weight, bias=None)

    #self.linear2_bn = torch.nn.BatchNorm1d(D_out)

    # nn.ReLU()

    self.out = nn.Softmax()

    

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """

    h_relu = self.linear1(x)

    y_pred = nn.functional.relu(self.linear2(h_relu))
    y_pred = nn.functional.relu(self.linear3(y_pred))
    y_pred = nn.functional.relu(self.linear4(y_pred))
    y_pred = nn.functional.relu(self.linear5(y_pred))

   

    y_pred = self.out(y_pred)

    return y_pred
    #return x


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, I, J, K, D_out = 50, 27338, 1000, 500,100, 50, 2
N, D_in, H, I, J, K, D_out = 50, 4221, 1000, 500, 100, 50, 2


# Create random Tensors to hold inputs and outputs, and wrap them in Variables
#lg("-------- {} time(s) CNN module start --------".format(times))
# model = TwoLayerNet(D_in, H, I, D_out)

#criterion = torch.nn.MSELoss(size_average=False)
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

    # torch 自带GPU运算
    # if torch.cuda.is_available():
    #     print('cuda is avaliable')
    #     x = x.cuda()
    #     y = y.cuda()
    #     x + y          # 在GPU上进行计算
    
    print("+++")
    # train_data = ProteinDataSet(
    train_data = ProteinDataSet(
        pd_dataFrame = training,
        transform = ToTensor()
    )
   

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # criterion = torch.nn.MSELoss(size_average=False)
    criterion = torch.nn.CrossEntropyLoss()

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
            b_y = Variable(y.float(), requires_grad=False)  # batch y            
            y_score = model(b_x)
            y_pred = torch.max(y_score, 1)[0]

            train_accuracy = sum(torch.max(y_score, 1)[1].data.squeeze() == b_y.data.long()) / float(
                b_y.size(0))

            train_loss = criterion(y_pred, b_y)
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

    
    print("\n {}".format(print_f_score(y_pred, b_y, average="weighted")))
    print("\n {}".format(accuracy_score(model.predict(test_dataset), test_dataset.labels)))


def bundle(times = 10):

    
    model = DNN(D_in, H, I, J, K, D_out)
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
        bundle()
