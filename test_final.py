import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from analysis import *
from train import *
# from dnn_main import *
from load_data import *
from scipy import interp
from settings import *
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable




def test_final(model, testing, args, index):
   
    print("=> loading weights from '/home/chunhui/yi/DNN/NewData/best_param/group_{}/best_accuracy.pth.tar'".format(index))

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    # assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = torch.load('/home/chunhui/yi/DNN/NewData/best_param/group_{}/best_accuracy.pth.tar'.format(index))
    model.load_state_dict(checkpoint['state_dict'])

    print("****")
    test_data = ProteinDataSet(
        pd_dataFrame=testing,
    )



    test_x = [test_data[i][0] for i in range(len(test_data))]
    # print(test_x)
    # test_x = Variable(torch.unsqueeze(torch.Tensor(test_x)))
    test_x = Variable(torch.Tensor(test_x))
    test_y = torch.from_numpy(test_data.labels)
    if args.cuda:
        test_x, test_y = test_x.cuda(), test_y.cuda()
    test_output = model(test_x[:])
    score = test_output.cpu().data.squeeze().numpy()

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

    # lg('\nTotal Accuracy: {:5.4f} \n    Pos Precision: {:5.4f} | Neg Precision: {:5.4f} \n    Pos Recall   : {:5.4f} | Neg Recall   : {:5.4f}'.format((TP+TN)/float(test_y.size(0)),TP/float(TP+FP),TN/float(TN+FN),TP/float(TP+FN),TN/float(TN+FP)))
    lg('\nTotal Accuracy: {:5.4f} \n    '.format((TP+TN)/float(test_y.size(0))))

    fpr = dict()
    tpr = dict()
    thresholds = dict()
    roc_auc = dict()

    for i in range(Class_N):
       

        fpr[i], tpr[i], thresholds[i] = roc_curve(labels[0], score[:,i], pos_label = i)
        # print("fpr[{}]:".format(i), fpr[i])
        # print("tpr[{}]:".format(i), tpr[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # print("auc[{}]:".format(i), roc_auc[i])
    lg('    Pos AUC      :{} | Neg AUC      :{}'.format(roc_auc[0],roc_auc[1]))
    #lg("\n-------- {} time(s) CNN module end --------\n\n".format(times))
    print("finished")
    return (fpr,tpr,roc_auc)

if __name__ == "__main__":
    model = TwoLayerNet(D_in, H, I, J, K, D_out)
    args = parser.parse_args()
    i = 0
    test_dataset = readTestData(
        label_data_path='{}{}'.format(args.test_data_folder, args.prefix_filename),
        index = i,
        total=args.groups,
    )

    test_final(model, test_dataset, args, i)