import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
from .drawRoc import drawFigureFrame, drawLine


def calPR(targets, predicts, pos_label):
    precision, recall, _ = precision_recall_curve(
        targets, predicts, pos_label=pos_label)  # precision recall threshold
    return precision, recall

def drawSinglePR(target, predict, pos_label=1, is_show=False, save_file=None):
    precision, recall = calPR(target, predict, pos_label)
    drawFigureFrame('Recall',
                    'Precision',
                    'PR Curve', xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    drawLine(precision, recall, label='PR curve', color='orange', lineWeight=2)

    plt.legend(loc="lower left")
    if save_file != None:
        plt.savefig(save_file, dpi=150)
    if is_show:
        plt.show()
    plt.close()


def drawMeanPR(targets, predicts, pos_label, is_show=False, save_file=None):
    recalls = []
    mean_precision = np.linspace(1, 0, 100)
    drawFigureFrame('Recall',
                    'Precision',
                    'PR Curve', xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    for i in range(len(targets)):
        # Compute PR curve and area the curve
        precision, recall = calPR(targets[i], predicts[i], pos_label)        
        recalls.append(interp(mean_precision, precision, recall))
        recalls[-1][0] = 0.0
        plt.plot(precision, recall, lw=1, alpha=0.3)
    # Mean PR
    mean_recall = np.mean(recalls, axis=0)
    mean_recall[-1] = 1.0
    plt.plot(mean_precision, mean_recall, color='b',
             label=r'Mean PR curve',
             lw=2, alpha=.8)

    # Standard area
    std_recall = np.std(recalls, axis=0)
    recalls_upper = np.minimum(mean_recall + std_recall, 1)
    recalls_lower = np.maximum(mean_recall - std_recall, 0)
    plt.fill_between(mean_precision, recalls_lower, recalls_upper, color='grey', alpha=.2,
                     label=r'Std PR curve')
    plt.legend(loc="lower left")
    if save_file != None:
        plt.savefig(save_file, dpi=150)
    if is_show:
        plt.show()
    plt.close()




def main():
    drawSinglePR([0, 1, 0, 1], [0.2, 0.4, 0.6, 0.8],
                  1, save_file="SinglePR.png")


if __name__ == '__main__':
    main()
