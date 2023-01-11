"""
    @file:              visualization.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:      This file contains simple functions related to visualization, mostly during training.
"""

import os
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor

from src.data.processing.tools import MaskType
from src.recording.constants import MEAN, STD


# Epochs progression figure name
EPOCHS_PROGRESSION_FIG: str = "epochs_progression.png"


def visualize_epoch_progression(
        train_history: List[List[float]],
        valid_history: List[List[float]],
        progression_type: List[str],
        path: str
) -> None:
    """
    Visualizes train and test loss histories over training epoch.

    Parameters
    ----------
    train_history : List[List[float]]
        A list of (E,) lists of loss values/evaluation metrics values across epochs where E is the number of epochs
    valid_history : List[List[float]]
        A list of (E,) list.
    progression_type : List[str]
        A list of string specifying the type of the progressions to visualize.
    path :
        Path where to save the plots.
    """
    plt.figure(figsize=(12, 8))

    # If there is only one plot to show (related to the loss)
    if len(train_history) == 1:

        x = range(len(train_history[0]))
        plt.plot(x, train_history[0], label=MaskType.TRAIN)
        plt.plot(x, valid_history[0], label=MaskType.VALID)

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(progression_type[0])

    # If there are multiple plots to show (one for the loss and one or many for the evaluation metric)
    else:
        for i in range(len(train_history)):

            nb_epochs = len(train_history[i])
            plt.subplot(1, len(train_history), i+1)
            plt.plot(range(nb_epochs), train_history[i], label=MaskType.TRAIN)
            if len(valid_history[i]) != 0:
                plt.plot(range(nb_epochs), valid_history[i], label=MaskType.VALID)

            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(progression_type[i])

    plt.tight_layout()
    plt.savefig(os.path.join(path, EPOCHS_PROGRESSION_FIG))
    plt.close()


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
