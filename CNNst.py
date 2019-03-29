import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from settings import *

class CNN_knnscore(nn.Module):
    def __init__(self):
        super(CNN_knnscore, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv1d(
                in_channels=CNN_P[0][0],              # input height
                out_channels=CNN_P[0][1],             # n_filters
                kernel_size=CNN_P[0][2],              # filter size
                stride=CNN_P[0][3],                   # filter movement/step
                padding=CNN_P[0][4],                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(kernel_size=CNN_P[0][5]),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
      
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv1d(CNN_P[1][0], CNN_P[1][1], CNN_P[1][2], CNN_P[1][3], CNN_P[1][4]),     # output shape (32, 7, 7)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(CNN_P[1][5]),                # output shape (32, 7, 7)
        )
        # self.out1 = nn.Linear(RS_Size, Class_N, True)   # fully connected layer, output 2 classes
        self.out1 = nn.Linear(384, 2, True)
        # self.out2 = nn.Softmax()
        self.out2 = nn.LogSoftmax(dim = 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out1(x)
        output = self.out2(output)
        return output    # return x for visualization