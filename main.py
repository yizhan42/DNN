import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from load_data import *
from settings import *
from CNNnd import *
from CNNst import *

from train import run_main
from test_final import runAndDraw

from sklearn.metrics import roc_curve, auc
from scipy import interp

def main(args):   
        model = CNN_multihot       
        run_main(model, args)
        runAndDraw(model, args)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args() 
    main(args)
