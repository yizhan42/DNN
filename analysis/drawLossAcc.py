from __future__ import division, print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def drawLossFigure(epoch, train_loss, valid_loss, save_path, is_print=True, is_save=False):
    plt.figure(figsize=(28, 14), dpi=80)
    plt.title('Train & Validation Loss Figure')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.ylim((0, 1.5)) # 写死为Y轴为 0 - 1.5
    plt.xticks(np.arange(epoch[0]-1, epoch[-1]+1, 20*(epoch[-1]//1000+1)))
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.plot(epoch, valid_loss, '-', label='validation')
    plt.plot(epoch, train_loss, '-', label='training')
    plt.legend()
    # plt.subplot(111).grid(True, which='major')
    ax = plt.gca()
    ax.grid(True)
    if(is_save):
        plt.savefig(save_path, dpi=80)
    if(is_print):
        plt.show()


def drawLossFigureFromFile(file_path, is_print=True, is_save=False):
    validation = []
    training = []
    epoches = []
    with open(file_path, 'r') as fp:
        for line in fp:
            if '|' in line and 'epoch' not in line:
                paras = line.split('|')
                if len(paras) == 7:
                    epoch, batch, valid_loss, train_loss, valid_acc, train_acc, learn_rate = paras
                else:
                    epoch, batch, valid_loss, train_loss, valid_acc, learn_rate = paras
                validation.append(float(valid_loss))
                training.append(float(train_loss))
                epoches.append(int(epoch))
    drawLossFigure(epoches, training, validation, '{}_loss.png'.format(
        file_path.split('.')[0]), is_print, is_save)


def drawLossFigureWhenCrossValidating(file_path_prefix, file_name='result.csv', start=0, end=10, is_print=True, is_save=False):
    for i in range(start, end):
        file_path = '{}_{}/{}'.format(file_path_prefix, i, file_name)
        drawLossFigureFromFile(file_path, is_print, is_save)


def drawLossFigureWhenCrossValidatingInSingleFile(file_path, is_print=True, is_save=False):
    validation = []
    training = []
    epoches = []
    count = 0
    with open(file_path, 'r') as fp:
        for line in fp:
            if '|' in line:
                if 'epoch' in line:
                    if epoches != []:
                        save_path = '{}_loss_{}.png'.format(
                            file_path.split('.')[0], count)
                        drawLossFigure(epoches, validation,
                                       training, save_path, is_print, is_save)
                        count += 1
                        validation = []
                        training = []
                        epoches = []
                else:
                    epoch, batch, valid_loss, train_loss, accuracy, learn_rate = line.split(
                        '|')
                    validation.append(float(valid_loss))
                    training.append(float(train_loss))
                    epoches.append(int(epoch))



def drawLossFigureWhenCrossValidatingInMultFiles(files, save_path, is_print=True, is_save=True):
    print("Drawing Loss and Acc Figures.....")
    vals = []
    trains = []
    accs = []
    # epoches = [x for x in range(1, 151)]
    for file_path in files:
        validation = []
        training = []
        acc = []
        # epoches = []
        # count = 0
        with open(file_path, 'r') as fp:
            for line in fp:
                if '|' in line and 'epoch' not in line:
                    line = line.split('|')
                    if len(line) == 7:
                        epoch, batch, valid_loss, train_loss, accuracy, train_acc, learn_rate = line
                    else:
                        epoch, batch, valid_loss, train_loss, accuracy, learn_rate = line
                    validation.append(float(valid_loss))
                    training.append(float(train_loss))
                    acc.append(float(accuracy))
                    # epoches.append(int(epoch))
        vals.append(np.array(validation))
        trains.append(np.array(training))
        accs.append(np.array(acc))
    epoches = [x for x in range(1, len(vals[0])+1)]
    vals = np.array(vals)
    trains = np.array(trains)
    accs = np.array(accs)
    # print(vals)

    mean_val = np.mean(vals, axis=0)
    std_val = np.std(vals, axis=0)
    val_upper = np.minimum(mean_val + std_val, 1.3)
    val_lower = np.maximum(mean_val - std_val, 0)

    mean_train = np.mean(trains, axis=0)
    std_train = np.std(trains, axis=0)
    train_upper = np.minimum(mean_train + std_train, 1.3)
    train_lower = np.maximum(mean_train - std_train, 0)

    # print(accs)
    mean_acc = np.mean(accs, axis=0)
    std_acc = np.std(accs, axis=0)
    acc_upper = np.minimum(mean_acc + std_acc, 100)
    acc_lower = np.maximum(mean_acc - std_acc, 0)

    # print(mean_val)
    plt.figure(figsize=(28, 14), dpi=80)
    plt.title('Train & Validation Loss Figure', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    # plt.ylim((0, 1))
    plt.yticks(np.arange(0, 2.1, 0.1), fontsize=20)
    plt.xticks(np.arange(epoches[0]-1, epoches[-1] +
                         1, len(epoches)/25), fontsize=20)
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    # print(epoches)
    plt.plot(epoches, mean_val, '-', label='Val mean loss')
    plt.plot(epoches, mean_train, '-', label='Tra mean loss')
    plt.fill_between(epoches, train_lower, train_upper, color='green', alpha=.2,
                     label=r'Tra std loss')
    plt.fill_between(epoches, val_lower, val_upper, color='red', alpha=.2,
                     label=r'Val std loss')
    plt.legend(fontsize=20, loc='lower left')
    # plt.subplot(111).grid(True, which='major')
    ax = plt.gca()
    ax.grid(True)
    if(is_save):
        print("Mean Loss Figure done!")
        plt.savefig('{}/mean_loss.png'.format(save_path), dpi=60)

    plt.figure(figsize=(28, 14), dpi=80)
    plt.title('Validation Accuracy Figure', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Acc', fontsize=20)
    # plt.ylim((50, 100))
    plt.yticks(np.arange(0, 1, 0.01), fontsize=20)
    plt.xticks(np.arange(epoches[0]-1, epoches[-1] +
                         1, len(epoches)/25), fontsize=20)
    # plt.yticks(np.arange(0.0, 1.0, 0.1))
    plt.plot(epoches, mean_acc, '-', label='Mean acc')
    # plt.plot(epoches, mean_train, '-', label='Mean-Training')
    plt.fill_between(epoches, acc_lower, acc_upper, color='green', alpha=.2,
                     label=r'Std acc')
    # plt.fill_between(epoches, val_lower, val_upper, color='blue', alpha=.2,
    #              label=r'Validation loss')
    plt.legend(fontsize=20, loc='upper left')
    # plt.subplot(111).grid(True, which='major')
    ax = plt.gca()
    ax.grid(True)
    if(is_save):
        print("Mean Acc Figure done!")
        plt.savefig('{}/mean_acc.png'.format(save_path), dpi=60)
    # if(is_print):
    #     plt.show()
    plt.close('all')

if __name__ == '__main__':

    # drawLossFigureFromFile('result/aws-1220/group_2/result.csv', True, True)
    # drawLossFigureFromFile('result/dlg-0227/3/result.csv', True, True)
    drawLossFigureWhenCrossValidating(
        'result/0425/group', 'result.csv', 0, 10, True, True)
