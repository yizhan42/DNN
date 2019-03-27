import argparse
import pickle
# from os import errno

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import auc, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from analysis import *
from FeatureSelectionData import * 
from load_data import *
from process_data import *
from settings import *
from test_final import *
from torch.autograd import Variable
from train import *

#
# """
# A fully-connected ReLU network with one hidden layer, trained to predict y from x
# by minimizing squared Euclidean distance.
# This implementation defines the model as a custom Module subclass. Whenever you
# want a model more complex than a simple sequence of existing Modules you will
# need to define your model this way.
# """





def main(args):
    # Run for each Cross Validation data
    save_folder = args.save_folder
    val_loss_folder = args.val_loss_folder
    val_accuracy_folder = args.val_accuracy_folder
    
    for i in range(args.start, args.end):

        if (args.start != args.end):
            args.save_folder = '{}/group_{}/'.format(save_folder, i)
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)
        print(args.save_folder)
        print('Loading data...')
        train_dataset, validation_dataset = readTrainingData(
            label_data_path='{}{}'.format(
                args.train_data_folder, args.prefix_filename),
            index=i,
            total=args.groups,
            # standard_length=args.length,
        )
        test_dataset = readTestData(
            label_data_path='{}{}'.format(
                args.test_data_folder, args.prefix_filename),
            index=i,
            total=args.groups,
        )
        model = TwoLayerNet(D_in, H, I, J, K, D_out)
        run(args, train_dataset, validation_dataset, test_dataset, model, i)

        if (args.start != args.end):
            args.val_loss_folder = '{}/loss_{}/'.format(val_loss_folder, i)
            if not os.path.exists(args.val_loss_folder):
                os.makedirs(args.val_loss_folder)

        if (args.start != args.end):
            args.val_accuracy_folder = '{}/accuracy_{}/'.format(
                val_accuracy_folder, i)
            if not os.path.exists(args.val_accuracy_folder):
                os.makedirs(args.val_accuracy_folder)

        plt_loss(args, train_loss_list[-100:], val_loss_list[-100:])
        drawAccuracy(
            args, train_accuracy_list[-100:], val_accuracy_list[-100:])


def run(args, train_dataset, validation_dataset, test_dataset, model, index):
    train(model, train_dataset, validation_dataset, args)
    test_final(model, test_dataset, args, index)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    main(args)
