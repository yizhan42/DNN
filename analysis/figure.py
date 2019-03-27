# coding=utf8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plt_roc(model,test,args):
    # model = TwoLayerNet(D_in, H, I, D_out)
    plt.switch_backend('agg')
    lg("\nModle:\n{}\n".format(model))  # net architecture

    tprs = [[], []]
    aucs = [[], []]
    mean_fpr = [np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
    times = 1

    # args = parser.parse_args()
    for i in range(times):
        fpr, tpr, roc_auc = test_final(model, test, args, i)
        for i in range(Class_N):
            tprs[i].append(interp(mean_fpr[i], fpr[i], tpr[i]))
            tprs[i][-1][0] = 0.0
            aucs[i].append(roc_auc[i])
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
    plt.savefig("/home/dlg/JL/yi/DNN/NewData/log/figure.png", dpi=150)
    plt.show()

   

def plt_loss(args, train_loss, val_loss):
    losses = {'train': [], 'validation': []}
    # train_loss = criterion(torch.max(model(Variable(train_data.properties)), 1)[0], b_y)
    # val_loss = criterion(torch.max(model(validation_x), 1)[0], validation_y)
    plt.figure(dpi=150)
    plt.title('Train & Validation Loss Figure')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')

    plt.legend()
    _ = plt.ylim()
    plt.savefig('{}figure_{}.png'.format(args.val_loss_folder, DATETIME), dpi=150)
    plt.show()

def drawAccuracy(args, train_accuracy_list, val_accuracy_list):
    plt.figure(dpi=150)
    plt.title('Train & Validation Accuracy Figure')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(train_accuracy_list, label = 'Train accuracy')
    plt.plot(val_accuracy_list, label = 'Validation accuracy')

    plt.legend()
    _ = plt.ylim()
    plt.savefig('{}figure_{}.png'.format(args.val_accuracy_folder, DATETIME),dpi = 150)
    plt.show()

def drawLossFigure(train_loss_list, val_loss_list, save_path, is_print = True, is_save = False):
    plt.figure(figsize=(28,14), dpi = 80)
    plt.title('Train & Validation Loss Figure')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim((0, 1.5))
    #plt.xticks(np.arange(epoch[0]-1, epoch[-1]+1, 20*(epoch[-1]//1000+1)))
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.plot(val_loss_list, '-', label='validation')
    plt.plot(train_loss_list, '-', label='training')
    plt.legend()
    # plt.subplot(111).grid(True, which='major')
    ax = plt.gca()
    ax.grid(True)
    if(is_save):
        plt.savefig("/home/dlg/JL/yi/DNN/NewData/train_val_loss/loss_png", dpi = 80)
    if(is_print):
        plt.show()


