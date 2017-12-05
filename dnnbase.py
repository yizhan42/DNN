import torch
from torch.autograd import Variable
from load_data import *
from settings import *

import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

import torch.nn.functional as F
"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""

class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    #self.H = nn.ReLU()

    # m = nn.ReLU()
    # input = autograd.Variable(torch.randn(2))
    # print(m(input))


    #H = Variable(H)
    self.linear2 = torch.nn.Linear(H, D_out)
    self.out = nn.Softmax()
    #self.linear2 = nn.Softmax()



    #oo = m.forward(H)

    # ii = torch.exp(torch.abs(torch.randn(10)))
    # m = nn.SoftMax()
    # oo = m:forward(ii)
    # gnuplot.plot({'Input', ii, '+-'}, {'Output', oo, '+-'})
    # gnuplot.grid(true)

    #softmax = nn.Softmax(dim= None)
    #self.linear2 = F.softmax()

  def forward(self, x):
    """
    In the forward function we accept a Variable of input data and we must return
    a Variable of output data. We can use Modules defined in the constructor as
    well as arbitrary operators on Variables.
    """
    # h_relu = self.linear1(x).clamp(min=0)
    # y_pred = self.linear2(h_relu)

    h_relu = nn.functional.relu(self.linear1(x))
    y_pred = self.linear2(h_relu)
    # print("start")
    # print(y_pred)
    y_pred = self.out(y_pred)
    # print(y_pred)
    # = Variable(y_pred)
    #y_pred = nn.functional.softmax(y_pred)
    #y_pred = nn.Softmax()
    #y_pred = nn.Softmax().forward(y_pred)
    #y_pred = nn.Softmax()
    #y_pred = F.softmax()
    return y_pred
    #return x


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 50, 175, 100, 2

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
#lg("-------- {} time(s) CNN module start --------".format(times))
def run(times=0, module = TwoLayerNet):
    torch.manual_seed(1)    # reproducible

    train, test = randomSplit(csv_file = CSV_FILE, pos_size=POS_SIZE, neg_size=NEG_SIZE, pick_rate=PICK_RATE)

    train_data = ProteinDataset(
        pd_dataFrame = train,
        transform = ToTensor()
    )
    test_data = ProteinDataset(
        pd_dataFrame = test,
    )

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    test_x = [test_data[i][0] for i in range(len(test_data))]
    # print(test_x)
    # test_x = Variable(torch.unsqueeze(torch.Tensor(test_x)))
    test_x = Variable(torch.Tensor(test_x))
    test_y = torch.from_numpy(test_data.labels)

    # for (x,y) in enumerate(train_loader):
    #     x = Variable(BATCH_SIZE,1)
    #     y = Variable(y)

    #

    #x = Variable(torch.randn(N, D_in))
    #y = Variable(torch.randn(N, D_out), requires_grad=False)
    #print(type(x), type(y))
    #print(x,y)

    # Construct our model by instantiating the class defined above
    model = TwoLayerNet(D_in, H, D_out)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):

            b_x = Variable(x)  # batch x
            b_y = Variable(y.float(), requires_grad=False)  # batch y
            #print(type(b_x), type(b_y))
            #print (b_x, b_y)

            # y_pred = model(b_x)[:,0]
            # print(y_pred)
            y_pred = model(b_x)
            y_pred = torch.max(y_pred, 1)[0]
            # print("1111")
            # print(y_pred)
            #print(b_x.size(), b_x)



      # Forward pass: Compute predicted y by passing x to the model
      #y_pred = model(x)

      # Compute and print loss
            loss = criterion(y_pred, b_y)
            #print(epoch, loss.data[0])

      # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # prediction = torch.max(F.softmax(y_pred), 1)[1]
            # print(prediction)

        if epoch % 100 == 0:
            # print(test_datax)
            pred_score = model(test_x)
            #F.softmax(pred_score)
            # print (pred_score)

            # test_output = pred_score[:,0]
            # print("score:")
            # print(test_output)
            #test_output = F.softmax((pred_score))[0]


            # print(torch.max(pred_score,1))
            pred_y = torch.max(pred_score,1)[1].data.squeeze()
            #print(pred_y)
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            lg('    Epoch: {:4d} | train loss: {:7.6f} | test accuracy: {:3.2f}'.format(epoch, loss.data[0], accuracy))


     # print predictions from test data
    #lg('\nFinial Testing {}:'.format(times))
    test_output = model(test_x[:])
    score = test_output.data.squeeze().numpy()

    # score = torch.max(test_output, 1)[0].data.squeeze()
    preds = torch.max(test_output,1)[1].data.squeeze()
    #print(preds)
    # print(test_data.labels,type(test_data.labels))
    labels = (test_data.labels,[1-label for label in test_data.labels])

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, pred in enumerate(preds):
        if(test_y[i] == 0):
            if(pred == test_y[i]):
                TN += 1
            else:
                FN += 1
        else:
            if(pred == test_y[i]):
                TP += 1
            else:
                FP += 1
        lg('    Predict:{} Real:{} - {}'.format(pred, test_y[i], 'hit' if pred == test_y[i] else 'miss'))

    lg('\nTotal Accuracy: {:5.4f} \n    Pos Precision: {:5.4f} | Neg Precision: {:5.4f} \n    Pos Recall   : {:5.4f} | Neg Recall   : {:5.4f}'.format((TP+TN)/float(test_y.size(0)),TP/(TP+FP+1),TN/(TN+FN+1),TP/(TP+FN+1),TN/(TN+FP+1)))

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    for i in range(Class_N):
        # print(preds[i].numpy(), score[:,i])
        # print(type(labels[i]), labels[i])
        # print(type(score[:, i]), score[:, i])

        fpr[i], tpr[i], thresholds[i] = roc_curve(labels[0], score[:,i], pos_label = i)
        # print("fpr[{}]:".format(i), fpr[i])
        # print("tpr[{}]:".format(i), tpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # print("auc[{}]:".format(i), roc_auc[i])
    lg('    Pos AUC      :{} | Neg AUC      :{}'.format(roc_auc[0],roc_auc[1]))
    #lg("\n-------- {} time(s) CNN module end --------\n\n".format(times))
    return (fpr,tpr,roc_auc)

def bundle(times = 10):
    plt.switch_backend('agg')
    lg("\nModle:\n{}\n".format(TwoLayerNet(D_in, H, D_out)))  # net architecture

    tprs = [[], []]
    aucs = [[], []]
    mean_fpr = [np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
    for i in range(times):
        fpr,tpr,roc_auc = run(i)
        for i in range(Class_N):
            tprs[i].append(interp(mean_fpr[i], fpr[i], tpr[i]))
            tprs[i][-1][0] = 0.0
            aucs[i].append(roc_auc[i])
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #          label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    for i in range(Class_N):
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs[i], axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr[i], mean_tpr)
        std_auc = np.std(aucs[i])
        plt.plot(mean_fpr[i], mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs[i], axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr[i], tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('{}figure_{}_{}.png'.format(LOG_PATH, DATETIME, i), dpi=150)
        plt.show()


def draw(fpr, tpr, roc_auc):

    plt.figure()
    lw = 2
    colors = ['darkorange', 'cornflowerblue']
    for i in range(Class_N):
        plt.plot(fpr[i], tpr[i], color=colors[i],
              lw=lw, label='ROC_AUC = %0.2f' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CNN ROC Curve')
    plt.legend(loc="lower right")
    # plt.show()
    # if LOGGED and IS_WRITABLE:
    plt.savefig("/home/dlg/JL/yi/DNN/log/figure.png", dpi=150)
    plt.show()

    # import scikitplot as skplt
    # print(labels[0], score.shape)
    # skplt.metrics.plot_roc_curve(labels[0], score)
    # plt.show()

if __name__ == "__main__":
    bundle(RUN_TIMES)