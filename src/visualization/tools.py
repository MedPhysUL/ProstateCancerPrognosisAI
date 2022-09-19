"""
    @file:              visualization.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:      This file contains simple functions related to visualization, mostly during training.
"""

import os
from typing import List

from matplotlib import pyplot as plt
from torch import Tensor

from src.data.processing.tools import MaskType


# Epochs progression figure name
EPOCHS_PROGRESSION_FIG: str = "epochs_progression.png"


def visualize_epoch_progression(
        train_history: List[Tensor],
        valid_history: List[Tensor],
        progression_type: List[str],
        path: str
) -> None:
    """
    Visualizes train and test loss histories over training epoch.

    Parameters
    ----------
    train_history : List[Tensor]
        A list of (E,) tensors where E is the number of epochs.
    valid_history : List[Tensor]
        A list of (E,) tensor.
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
            plt.subplot(1, 2, i+1)
            plt.plot(range(nb_epochs), train_history[i], label=MaskType.TRAIN)
            if len(valid_history[i]) != 0:
                plt.plot(range(nb_epochs), valid_history[i], label=MaskType.VALID)

            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel(progression_type[i])

    plt.tight_layout()
    plt.savefig(os.path.join(path, EPOCHS_PROGRESSION_FIG))
    plt.close()
