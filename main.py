import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from load_data import *
from settings import *

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
# Module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(
                in_channels=CNN_P[0][0],              # input height
                out_channels=CNN_P[0][1],             # n_filters
                kernel_size=CNN_P[0][2],              # filter size
                stride=CNN_P[0][3],                   # filter movement/step
                padding=CNN_P[0][4],                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=CNN_P[0][5]),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        # self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
        #     nn.Conv2d(CNN_P[1][0], CNN_P[1][1], CNN_P[1][2], CNN_P[1][3], CNN_P[1][4]),     # output shape (32, 7, 7)
        #     nn.ReLU(),                      # activation
        #     nn.MaxPool2d(CNN_P[1][5]),                # output shape (32, 7, 7)
        # )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(CNN_P[1][0], CNN_P[1][1], CNN_P[1][2], CNN_P[1][3], CNN_P[1][4]),     # output shape (32, 7, 7)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(CNN_P[1][5]),                # output shape (32, 7, 7)
        )
        # self.out1 = nn.Linear(RS_Size, Class_N, True)   # fully connected layer, output 2 classes
        self.out1 = nn.Linear(RS_Size, 2, True)
        # self.out2 = nn.Softmax()
        self.out2 = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out1(x)
        output = self.out2(output)
        return output, x    # return x for visualization

def run(times=0, module = CNN):
    lg("-------- {} time(s) CNN module start --------".format(times))
    # torch.manual_seed(1)    # reproducible

    train, test = randomSplit(csv_file = CSV_FILE, pos_size=POS_SIZE, neg_size=NEG_SIZE, pick_rate=PICK_RATE)

    train_data = ProteinDataset(
        pd_dataFrame = train,
        transform = ToTensor()
    )
    test_data = ProteinDataset(
        pd_dataFrame = test,
    )
    # Data Loader for easy mini-batch return in training, the Matrix batch shape will be (50, 1, 14, 14)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Bundle propertes and labels of test-data separately
    test_x = [test_data[i][0] for i in range(len(test_data))]
    test_x = Variable(torch.unsqueeze(torch.Tensor(test_x), dim=1))
    test_y = torch.from_numpy(test_data.labels)

    cnn = CNN()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
    # loss_func = nn.NLLLoss()
    # training and testing
    lg('Training {}:'.format(times))
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
            b_x = Variable(x)   # batch x
            b_y = Variable(y)   # batch y
            output = cnn(b_x)[0]            # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

        if epoch % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            lg('sum = {:4d}\n'.format(sum(pred_y == test_y)))
            lg('test size = {:4f}\n'.format(float(test_y.size(0))))
            
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            lg('accuracy = {:7.6f} \n'.format(accuracy))
            lg('    Epoch: {:4d} | train loss: {:7.6f} | test accuracy: {:3.2f}'.format(epoch, loss.data.item(), accuracy))

    # print predictions from test data
    lg('\nFinial Testing {}:'.format(times))
    test_output, _ = cnn(test_x[:])
    score = test_output.data.squeeze().numpy()
    # score = torch.max(test_output, 1)[0].data.squeeze()
    preds = (torch.max(test_output, 1)[1].data.squeeze(), torch.min(test_output, 1)[1].data.squeeze())
    # print(test_data.labels,type(test_data.labels))
    labels = (test_data.labels,[1-label for label in test_data.labels])
    # print(score)
    # print(labels)
    # print(test_y,type(test_y.numpy()))
    # print((score[:,0]),(preds[0].numpy()))
    print(test_output)
    # for a,b in test_output:
    #     print(a+b)
    # print0(help(test_output))
    # accuracy = sum(pred_y == test_y) / float(test_y.size(0))
    # print(test_output.requires_grad)#,torch.max(test_output, 1))
    # print("_:")
    # print(_)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, pred in enumerate(preds[0]):
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

    lg('\nTotal Accuracy: {:5.4f} \n    Pos Precision: {:5.4f} | Neg Precision: {:5.4f} \n    Pos Recall   : {:5.4f} | Neg Recall   : {:5.4f}'.format((TP+TN)/float(test_y.size(0)),TP/(TP+FP),TN/(TN+FN),TP/(TP+FN),TN/(TN+FP)))



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
    lg("\n-------- {} time(s) CNN module end --------\n\n".format(times))
    return (fpr,tpr,roc_auc)

def bundle(times = 10):
    # plt.switch_backend('agg')
    lg("\nModle:\n{}\n".format(CNN()))  # net architecture

    tprs = [[], []]
    aucs = [[], []]
    mean_fpr = [np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
    for i in range(times):
        fpr,tpr,roc_auc = run(i)
        for i in range(Class_N):
            tprs[i].append(interp(mean_fpr[i], fpr[i], tpr[i]))
            tprs[i][-1][0] = 0.0
            aucs[i].append(roc_auc[i])
            lg("\nClass {}:\n    tpr:{}\n    fpr:{}\n    Auc:{}".format(i, tpr[i], fpr[i], roc_auc[i]))
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
    plt.savefig(FIGURE_FILE, dpi=150)
    plt.show()

    # import scikitplot as skplt
    # print(labels[0], score.shape)
    # skplt.metrics.plot_roc_curve(labels[0], score)
    # plt.show()

if __name__ == "__main__":
    bundle(RUN_TIMES)
