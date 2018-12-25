'''
Project: 
Wind Gust Detection using Physical Sensors in Quadcopters

Author: 
Suwen Gu and Menghao Lin

Description:
This file contains some of the utility functions for 
    1. visualizing data in frequency-domain
    2. plotting confusion matrix
    3. plotting learning curve.

'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.utils import safe_indexing
import matplotlib.patches as mpatches
from scipy import signal
import seaborn as sns

def plot_in_frequency_domain(data, label, sensor):
    """
    Generates a simple plot of data in frequency-domain for all axes.

    Parameters
    ----------
    data: DataFrame
        the raw data
    label: str
        the class label
    sensor: str
        the name of the senor
    """

    # break the data set into three subsets
    x = data[sensor+'.x']
    y = data[sensor+'.y']
    z = data[sensor+'.z']
    f = plt.figure(figsize=(15, 3))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)
    
    # use Fourier Transform to get Fourier coefficients
    # and visualize data using power spectral density plot
    x_fft = np.fft.fft(x)
    f_x = np.fft.fftfreq(x.shape[0])
    Pxx_den_x = (x_fft*np.conj(x_fft))/x_fft.shape[0]
    ax.set_title(sensor + ': X axis, label: ' + str(label))
    ax.plot(f_x, Pxx_den_x)
    
    y_fft = np.fft.fft(y)
    f_y = np.fft.fftfreq(y.shape[0])
    Pxx_den_y = (y_fft*np.conj(y_fft))/y_fft.shape[0]
    ax2.set_title(sensor + ': Y axis, label: ' + str(label))
    ax2.plot(f_y, Pxx_den_y)
    
    z_fft = np.fft.fft(z)
    f_z = np.fft.fftfreq(z.shape[0])
    Pxx_den_z = (z_fft*np.conj(z_fft))/z_fft.shape[0]
    ax3.set_title(sensor + ': Z axis, label: ' + str(label))
    ax3.plot(f_z, Pxx_den_z)


def plot_confusion_matrix(confusion_matrix):
    """
    Uses heatmap of Seaborn library to visualize the confusion matrix

    Parameters
    ----------
    consufsion_matrix: DataFrame
        values in the confusion matrix
    """

    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.show()


"""
- Title of program: Plotting Learning Curves
- Author: Scikit-learn 
- Date: 12/24/2018
- Code Version: -
- Type: Source Code
- Availability: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt