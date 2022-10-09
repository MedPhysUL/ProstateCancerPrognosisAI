"""
    @file:              early_stopper.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the EarlyStopper class.
"""

from abc import ABC, abstractmethod
from enum import Enum
from os import path, remove
from typing import List, OrderedDict, Union
from uuid import uuid4

import numpy as np
from torch import load, save, Tensor
from torch.nn import Module

from src.utils.multi_task_losses import MultiTaskLoss
from src.utils.score_metrics import Direction
from src.utils.tasks import Task


class EarlyStopperType(Enum):
    """
    Custom enum for early stopper types.
    """
    METRIC = "Metric"
    MULTITASK_LOSS = "MultiTaskLoss"

    def __iter__(self):
        return iter([self.METRIC, self.MULTITASK_LOSS])


class EarlyStopper(ABC):

    def __init__(
            self,
            path_to_model: str,
            patience: int
    ) -> None:
        """
        Sets protected attributes of early stopper.

        Parameters
        ----------
        path_to_model : str
            Path to save model.
        patience : int
            Number of consecutive epochs without improvement allowed.
        """
        # Set public attribute
        self.patience = patience
        self.early_stop = False
        self.counter = 0
        self.best_model = None
        self.path_to_model = path_to_model
        self.file_path = path.join(path_to_model, f"{uuid4()}.pt")

    @property
    @abstractmethod
    def early_stopper_type(self) -> EarlyStopperType:
        raise NotImplementedError

    @abstractmethod
    def __call__(
            self,
            val_score: Union[float, List[float]],
            model: Module
    ) -> None:
        """
        Compares current best validation score against the given one and updates the EarlyStopper's attributes.

        Parameters
        ----------
        val_score : Union[float, List[float]]
            New validation loss or new validation scores for each tasks.
        model : Module
            Current model for which we've seen the score.
        """
        raise NotImplementedError

    @abstractmethod
    def print_early_stopping_message(
            self,
            epoch: int
    ) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        epoch : int
            Number of training epochs done
        """
        raise NotImplementedError

    def remove_checkpoint(
            self
    ) -> None:
        """
        Removes the checkpoint file
        """
        remove(self.file_path)

    def reset(
            self
    ) -> None:
        """
        Reset early stopper.
        """
        self.remove_checkpoint()
        self.__init__(self.path_to_model, self.patience)

    def get_best_params(
            self
    ) -> OrderedDict[str, Tensor]:
        """
        Returns the saved best parameters.

        Returns
        -------
        model_state : dict
            Model state.
        """
        return load(self.file_path)


class MetricEarlyStopper(EarlyStopper):

    def __init__(
            self,
            path_to_model: str,
            patience: int
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.
        Parameters
        ----------
        path_to_model : str
            Path to save model.
        patience : int
            Number of consecutive epochs without improvement allowed.
        """
        super().__init__(path_to_model=path_to_model, patience=patience)

        self._tasks = []
        self._best_val_scores = []

    @property
    def best_val_scores(self) -> List[float]:
        return self._best_val_scores

    @property
    def early_stopper_type(self) -> EarlyStopperType:
        return EarlyStopperType.METRIC

    @property
    def tasks(self) -> List[Task]:
        return self._tasks

    @tasks.setter
    def tasks(self, tasks: List[Task]):
        if not self._best_val_scores:
            self._best_val_scores = [
                np.inf if t.optimization_metric.direction == Direction.MINIMIZE.value else -np.inf for t in tasks
            ]

        self._tasks = tasks

    def print_early_stopping_message(
            self,
            epoch: int
    ) -> None:
        """
        Prints a message when early stopping occurs.
        Parameters
        ----------
        epoch : int
            Number of training epochs done
        """
        print(f"\nEarly stopping occurred at epoch {epoch} with best_epoch = {epoch - self.patience}")

        for score, task in zip(self.best_val_scores, self._tasks):
            print(f"Task ({task.name}) (metric {task.optimization_metric.name}), Score :{round(score, 4)}")

    def __call__(
            self,
            val_scores: List[float],
            model: Module
    ) -> None:
        """
        Compares current best validation score against the given one and updates the EarlyStopper's attributes.
        Parameters
        ----------
        val_scores : List[float]
            New validation scores for each tasks.
        model : Module
            Current model for which we've seen the score.
        """
        new_scores_is_better = []
        for i, task, val_score, best_score in enumerate(zip(self._tasks, val_scores, self._best_val_scores)):
            if task.optimization_metric.direction == Direction.MINIMIZE.value:
                new_score_is_better = val_score < best_score
            else:
                new_score_is_better = val_score > best_score

            new_scores_is_better.append(new_score_is_better)

            if new_score_is_better:
                self._best_val_scores[i] = val_score

        # if all the scores are worst than the best scores, we increment the counter
        if not any(new_scores_is_better):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            save(model.state_dict(), self.file_path)
            self.counter = 0


class MultiTaskLossEarlyStopper(EarlyStopper):

    def __init__(
            self,
            path_to_model: str,
            patience: int
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        path_to_model : str
            Path to save model.
        patience : int
            Number of consecutive epochs without improvement allowed.
        """
        super().__init__(path_to_model=path_to_model, patience=patience)

        self._criterion = None
        self.is_better = lambda x, y: x < y
        self._best_val_loss = np.inf

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    @property
    def criterion(self) -> MultiTaskLoss:
        return self._criterion

    @criterion.setter
    def criterion(self, criterion: MultiTaskLoss):
        self._criterion = criterion

    @property
    def early_stopper_type(self) -> EarlyStopperType:
        return EarlyStopperType.LOSS

    def print_early_stopping_message(
            self,
            epoch: int
    ) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        epoch : int
            Number of training epochs done
        """
        print(f"\nEarly stopping occurred at epoch {epoch} with best_epoch = {epoch - self.patience}")
        print(f"Criterion {self.criterion.name}), Loss :{round(self.best_val_loss, 4)}")

    def __call__(
            self,
            val_loss: float,
            model: Module
    ) -> None:
        """
        Compares current best validation loss against the given one and updates the EarlyStopper's attributes.

        Parameters
        ----------
        val_loss : float
            New validation loss.
        model : Module
            Current model for which we've seen the score.
        """
        # if the score is worst than the best score we increment the counter
        if not self.is_better(val_loss, self._best_val_loss):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self.early_stop = True

        # if the score is better than the best score saved we update the best model
        else:
            self._best_val_loss = val_loss
            save(model.state_dict(), self.file_path)
            self.counter = 0
