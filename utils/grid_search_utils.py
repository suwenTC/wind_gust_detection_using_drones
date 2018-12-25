'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
This file is created to perform grid search and print out 
the cross validation results in a more user-friendly and readable format.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score

def report_model_score(clf, X, y, cv=3):
    """ Generates and outputs cv score

    Parameters
    ----------
    clf: object
        sklearn classifier
    X: DataFrame or ndarray
        training data
    y: ndarray
        training data labels

    Returns:
    ----------
    """
    scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1)
    print('Model Report: ')
    print('Mean cv score: {:.3f} +/- std: {:.3f}'.format(np.mean(scores), np.std(scores)))
    
def model_selection(clf, param_grid, X , y):
    """ Performs grid search cross validation and returns the results.

    Parameters
    ----------
    clf: object
        sklearn classifier
    param_grid: dict
        the list of hyper-parameters to be tuned
    X: DataFrame or ndarray
        training data
    y: ndarray
        training data labels

    Returns:
    ----------
    object:
        the GridSearchCV object after fitting with training data
    """
    cv = KFold(n_splits=3, shuffle=True, random_state=13)
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(X, y)
    return grid_search

def report(grid_scores, n_top):
    """ Prints out the scores of the top-n best estimators
    
    The top-n best estimators are found after grid search.

    Parameters
    ----------
    grid_scores:  list of named tuples
       the scores for all parameter combinations in param_grid. 
    
    n_top: int
        the number of score being printed
    Returns:
    ----------
    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f}, Standard Deviation: {1:.4f}".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")