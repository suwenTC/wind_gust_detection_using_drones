'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
You do not need this file to generate a multiclass confusion matrix!

This file is used to generate multiclass confusion matrices.

'''

import math
import pandas as pd
import numpy as np
from sklearn.utils import safe_indexing

def _get_confusion_matrix_helper(counts, class_labels):
    """ Helper function, generates a confusion matrix

    Parameters
    ----------
    counts: dictionary
        the list of true labels and predicted labels
    class_labels: list
        the list of all possible class labels

    Returns:
    ----------
    DataFrame
        Multiclass confusion matrix
    """

    for key, item in counts.items():
        if len(item) < len(class_labels):
            for i in range(len(class_labels) - len(item)):
                counts[key] = np.append(counts[key], 0)

    index = ["actual_" + str(c) for c in class_labels]

    confusion_matrix = pd.DataFrame(data=counts, index=index)
    return confusion_matrix.T

def get_confusion_matrix(y_pred, y_true):
    """ Generates a confusion matrix
        
    Aggregates predicted and true labels into one dictionary

    Parameters
    ----------
    y_pred: ndarray
        the list of predicted labels
    y_true: ndarray
        the list of true labels

    Returns:
    ----------
    DataFrame
        Multiclass confusion matrix
    """
    counts = {}
    class_labels = np.unique(y_true)
    
    # according each label, count TP, FP, TN, and FN
    for label in class_labels:
        idx = np.flatnonzero(y_pred == label)
        counts[str("predicted_")+str(label)] = np.bincount(safe_indexing(y_true, idx))
 
    confusion_matrix = _get_confusion_matrix_helper(counts, class_labels)

    return confusion_matrix