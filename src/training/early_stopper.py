"""
    @file:              early_stopper.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the EarlyStopper class.
"""

from os import path, remove
from typing import OrderedDict
from uuid import uuid4

import numpy as np
from torch import load, save, tensor
from torch.nn import Module

from src.utils.score_metrics import Direction


class EarlyStopper:

    def __init__(
            self,
            direction: Direction,
            path_to_model: str,
            patience: int
    ) -> None:
        """
        Sets protected attributes of early stopper and define comparison method according to the given direction.

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        path_to_model : str
            Path to save model.
        patience: int
            Number of epochs without improvement.
        """
        # Set public attribute
        self.patience = patience
        self.early_stop = False
        self.counter = 0
        self.best_model = None
        self.file_path = path.join(path_to_model, f"{uuid4()}.pt")

        # Set comparison method
        if direction == Direction.MINIMIZE:
            self.best_val_score = np.inf
            self.is_better = lambda x, y: x < y

        elif direction == Direction.MAXIMIZE:
            self.best_val_score = -np.inf
            self.is_better = lambda x, y: x > y
        else:
            raise ValueError(f'direction must be in {list(Direction)}')

    def __call__(
            self,
            val_score: float,
            model: Module
    ) -> None:
        """
        Compares current best validation score against the given one and updates the EarlyStopper's attributes.

        Parameters
        ----------
        val_score : float
            New validation score.
        model : Module
            Current model for which we've seen the score.
        """
        # if the score is worst than the best score we increment the counter
        if not self.is_better(val_score, self.best_val_score):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            self.best_val_score = val_score
            save(model.state_dict(), self.file_path)
            self.counter = 0

    def remove_checkpoint(
            self
    ) -> None:
        """
        Removes the checkpoint file
        """
        remove(self.file_path)

    def get_best_params(
            self
    ) -> OrderedDict[str, tensor]:
        """
        Returns the saved best parameters.

        Returns
        -------
        model_state : dict
            Model state.
        """
        return load(self.file_path)
