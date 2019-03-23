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

from train import *
from test_final import * 

from sklearn.metrics import roc_curve, auc
from scipy import interp

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
        model = CNN_multihot()
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

def temp(module, args, times=0):
    lg("-------- {} time(s) CNN module start --------".format(times))
    # torch.manual_seed(1)    # reproducible

    train, test = randomSplit(csv_file = CSV_FILE, pos_size=POS_SIZE, neg_size=NEG_SIZE, pick_rate=PICK_RATE)

    # train_data = ProteinDataset(
    #     pd_dataFrame = train,
    #     transform = ToTensor()
    # )
    # test_data = ProteinDataset(
    #     pd_dataFrame = test,
    # )

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
    # Data Loader for easy mini-batch return in training, the Matrix batch shape will be (50, 1, 14, 14)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Bundle propertes and labels of test-data separately
    test_x = [test_dataset[i][0] for i in range(len(test_dataset))]
    print(test_x)
    test_x = Variable(torch.unsqueeze(torch.Tensor(test_x), dim=1))
    print("***")
    print(test_x)
    test_y = torch.from_numpy(test_dataset.labels)

    cnn = module

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
            # lg('sum = {:4d}\n'.format(sum(pred_y == test_y)))
            # lg('test size = {:4f}\n'.format(float(test_y.size(0))))
            
            accuracy = float(sum(pred_y == test_y)) / float(test_y.size(0))
            # lg('accuracy = {:7.6f} \n'.format(accuracy))
            lg('    Epoch: {:4d} | train loss: {:7.6f} | test accuracy: {:3.2f}'.format(epoch, loss.data.item(), accuracy))

    # print predictions from test data
    lg('\nFinial Testing {}:'.format(times))
    test_output, _ = cnn(test_x[:])
    score = test_output.data.squeeze().numpy()
    # score = torch.max(test_output, 1)[0].data.squeeze()
    preds = (torch.max(test_output, 1)[1].data.squeeze(), torch.min(test_output, 1)[1].data.squeeze())
    # print(test_data.labels,type(test_data.labels))
    labels = (test_dataset.labels,[1-label for label in test_dataset.labels])
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

def bundle(module, times = 10):
    # plt.switch_backend('agg')
    lg("\nModle:\n{}\n".format(module))  # net architecture

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
    bundle(CNN_knnscore, RUN_TIMES)
