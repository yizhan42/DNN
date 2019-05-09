import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def calRocAuc(targets, predicts, pos_label):
    fpr, tpr, _ = roc_curve(
        targets, predicts, pos_label=pos_label)  # fpr tpr threshold
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def drawFigureFrame(xlabel='x', ylabel='y', title='Figure',
                    xlim=[0.1, 1.0], ylim=[0.0, 1.05]):
    plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    # plt.yticks(np.arange(0, 1.4, 0.1), fontsize=20)
    # plt.xticks(np.arange(epoches[0]-1, epoches[-1]+1, 10), fontsize=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def drawLine(x_axis, y_axis, label='line', color='orange', lineWeight=1):
    plt.plot(x_axis, y_axis, color=color, lw=lineWeight, label=label)


def drawSingleRoc(target, predict, pos_label=1, is_show=False, save_file=None):
    fpr, tpr, auc = calRocAuc(target, predict, pos_label)
    drawFigureFrame('False Positive Rate (1-Specificity)',
                    'True Positive Rate (Sensitivity)',
                    'Roc Curve with Auc', xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    drawLine(fpr, tpr, label='ROC (AUC={:.2f})'.format(
        auc), color='orange', lineWeight=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="lucky")
    plt.legend(loc="lower right")
    if save_file != None:
        plt.savefig(save_file, dpi=150)
    if is_show:
        plt.show()
    plt.close()


def drawMeanRoc(targets, predicts, pos_label, is_show=False, save_file=None):
    print("targets:",targets)
    print("predicts:", predicts)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    print(2)
    drawFigureFrame('False Positive Rate (1-Specificity)',
                    'True Positive Rate (Sensitivity)',
                    'Roc Curve with Auc', xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    
    print(2)
    for i in range(len(targets)):
        # Compute ROC curve and area the curve
        fpr, tpr, roc_auc = calRocAuc(targets[i], predicts[i], pos_label)        
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)
                 #, label='ROC fold %d (AUC=%0.2f)' % (i, roc_auc))
    print(3)
    # Luck Line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='r', label='Luck', alpha=.8)
    print(4)
    # Mean roc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    print(5)
    # Standard area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'Std Roc')
    plt.legend(loc="lower right")
    if save_file != None:
        plt.savefig(save_file, dpi=150)
    if is_show:
        plt.show()
    plt.close()

def main():
    drawSingleRoc([0, 1, 0, 1], [0.2, 0.4, 0.6, 0.8],
                  1, save_file="SingleRoc.png")


if __name__ == '__main__':
    main()
