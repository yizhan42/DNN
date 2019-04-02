import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from settings import *
# import math
D_in, H, I, J, K, D_out = 4221, 1000, 500, 100, 50, 2

class DNN_multihot(torch.nn.Module):
	def __init__(self, D_in=D_in, H = H, I = I, J = J, K = K, D_out = D_out):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(DNN_multihot, self).__init__()

		#self.input = torch.nn.Dropout(p = 0.75)
		self.linear1 = torch.nn.Linear(D_in, H)
		# torch.manual_seed(1)
		# nn.init.xavier_normal(self.linear1.weight)
		# print(self.linear1.weight.data)
		# std = math.sqrt(2) / math.sqrt(7.)
		# self.linear1.weight.data.normal_(0, std)


		#self.linear1_bn = torch.nn.BatchNorm1d(H)
		
		#self.H = nn.ReLU()

		# m = nn.ReLU()
		# input = autograd.Variable(torch.randn(2))
		# print(m(input))


		#H = Variable(H)
		self.linear2 = torch.nn.Linear(H, I)
		self.dropout = nn.Dropout(p=0.5)
		# torch.manual_seed(1)
		# init.xavier_normal(self.linear2.weight)
		# print(self.linear2.weight.data)
		# std = math.sqrt(2) / math.sqrt(7.)
		# self.linear2.weight.data.normal_(0, std)
		# nn.Dropout(p = 0.75)


		self.linear3 = torch.nn.Linear(I, J)
		# torch.manual_seed(1)
		# init.xavier_normal(self.linear3.weight)
		# print(self.linear3.weight.data)
		# std = math.sqrt(2) / math.sqrt(7.)
		# self.linear3.weight.data.normal_(0, std)

		self.linear4 = torch.nn.Linear(J, K)
		self.linear5 = torch.nn.Linear(K, D_out)
		# torch.manual_seed(1)
		# init.xavier_normal(self.linear4.weight)
		# print(self.linear4.weight.data)
		# std = math.sqrt(2) / math.sqrt(7.)
		# self.linear4.weight.data.normal_(0, std)
		# self.dropout = nn.Dropout(p=0.25)
		# self.linear_normal(self.linear5.weight)
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
		self.dp = nn.Dropout(p = 0.75)
		self.out = nn.LogSoftmax(dim=1)

	def forward(self, x):
			"""
			In the forward function we accept a Variable of input data and we must return
			a Variable of output data. We can use Modules defined in the constructor as
			well as arbitrary operators on Variables.
			"""

			h_relu = self.linear1(x)

			y_pred = nn.functional.relu(self.linear2(h_relu))
			y_pred = self.dropout(y_pred)
			y_pred = nn.functional.relu(self.linear3(y_pred))
			y_pred = nn.functional.relu(self.linear4(y_pred))
			# y_pred = self.dropout(y_pred)
			y_pred = self.linear5(y_pred)
			y_pred = self.dp(y_pred)
			y_pred = self.out(y_pred)
			# print(y_pred)
			return y_pred
			#return x
