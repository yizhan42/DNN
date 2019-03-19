# coding=utf8

from math import sqrt

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(target, predict):
    """
    Evalute single result
    """
    cm = confusion_matrix(target, predict).ravel()
    tn, fp, fn, tp = cm if len(cm) == 4 else (0, 0, 0, cm[0])
    total = len(target)
    accuracy = (tp+tn)/total
    sens = np.float64(tp)/(tp+fn)  # true positive rate, recall, sensitivity
    spec = np.float64(tn)/(fp+tn)  # true negative rat, specificity
    ppv = np.float64(tp)/(tp+fp)  # positive predictive value, precision
    npv = np.float64(tn)/(tn+fn)  # negative predictive value
    f1 = np.float64(2*tp)/(2*tp+fp+fn)  # positive f1 score
    mcc = np.float64(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    accuracies = [np.float64(tn)/(tn+fn), np.float64(tp)/(tp+fp)]
    return np.array([accuracy, sens, spec, ppv, npv, f1, mcc]), accuracies
