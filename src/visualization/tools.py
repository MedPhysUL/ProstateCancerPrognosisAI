"""
    @file:              visualization.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:      This file contains simple functions related to visualization, mostly during training.
"""

from typing import Dict

from matplotlib import pyplot as plt
import numpy as np

from ..recording.constants import MEAN, STD


def visualize_importance(
        data: Dict[str, Dict[str, float]],
        figure_title: str,
        filename: str
) -> None:
    """
    Creates a bar plot with mean and standard deviations of variable importance contained within the dictionary.

    Parameters
    ----------
    data : Dict[str, Dict[str, float]]
        Dictionary with variable name as keys and "mean" and "std" as values.
    figure_title : str
        Name appearing over the plot.
    filename : str
        Name of the file in which the figure is saved.
    """
    # We initialize three lists for the values, the errors, and the labels
    means, stds, labels = [], [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        mean = data[key][MEAN]
        if mean >= 0.01:
            means.append(mean)
            stds.append(data[key][STD])
            labels.append(key)

    # We sort the list according to their values
    sorted_means = sorted(means)
    sorted_labels = sorted(labels, key=lambda x: means[labels.index(x)])
    sorted_stds = sorted(stds, key=lambda x: means[stds.index(x)])

    # We build the plot
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.barh(y_pos, sorted_means, xerr=sorted_stds, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Importance')
    ax.set_title(figure_title)

    # We save the plot
    plt.savefig(filename)
    plt.close()


def visualize_scaled_importance(
        data: Dict[str, Dict[str, float]],
        figure_title: str,
        filename: str
) -> None:
    """
    Creates a bar plot with mean and standard deviations of variable importance contained within the dictionary.

    Parameters
    ----------
    data : Dict[str, Dict[str, float]]
        Dictionary with variable name as keys and "mean" and "std" as values.
    figure_title : str
        Name appearing over the plot.
    filename : str
        Name of the file in which the figure is saved.
    """
    # We initialize two lists for the scaled values and the labels
    scaled_imp, labels = [], []

    # We collect the data of each hyperparameter importance
    for key in data.keys():
        mean = data[key][MEAN]
        if mean >= 0.01:
            scaled_imp.append(mean/(data[key][STD] + 0.001))
            labels.append(key)

    # We sort the list according values
    sorted_scaled_imp = sorted(scaled_imp)
    sorted_labels = sorted(labels, key=lambda x: scaled_imp[labels.index(x)])

    # We build the plot
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.barh(y_pos, sorted_scaled_imp, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Importance')
    ax.set_title(figure_title)

    # We save the plot
    plt.savefig(filename)
    plt.close()
