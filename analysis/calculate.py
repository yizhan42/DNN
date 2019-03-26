# coding=utf8

from math import sqrt

import numpy as np
from sklearn.metrics import confusion_matrix


def evaluate(target, predict):
    try:
        target =  target.cpu()
        predict = predict.cpu()
    except Exception as e:
        print(e)
    cm = confusion_matrix(target, predict).ravel()
    # print(cm)
    # tn, fp, fn, tp = cm
    tn, fp, fn, tp = cm if len(cm) == 4 else (0, 0, 0, cm[0])
    # print(tn, fp, fn, tp)
    total = len(target)
    accuracy = (tp+tn)/total
    
    sens = np.float64(tp)/(tp+fn) if tp+fn != 0 else 0 # true positive rate, recall, sensitivity
    spec = np.float64(tn)/(fp+tn)  if fp+tn != 0 else 0  # true negative rat, specificity
    ppv = np.float64(tp)/(tp+fp)  if tp+fp != 0 else 0# positive predictive value, precision
    npv = np.float64(tn)/(tn+fn)  if tn+fn != 0 else 0# negative predictive value
    f1 = np.float64(2*tp)/(2*tp+fp+fn)  # positive f1 score
    mcc = np.float64(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) != 0 else 0
    accuracies = [sens, spec]
    return np.array([accuracy, sens, spec, ppv, npv, f1, mcc]), accuracies