"""
    @file:              early_stopping.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 12/2022

    @Description:       This file is used to define the 'MetricEarlyStopping' and 'MultiTaskLossEarlyStopping' callback.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.callbacks.callback import Callback, Priority
from src.utils.score_metrics import Direction
from src.utils.tasks import Task


class BaseEarlyStopping(ABC, Callback):
    """
    Base class for early stopping.
    """

    def __init__(
            self,
            name: str,
            patience: int,
            tolerance: float,
            **kwargs
    ) -> None:
        """
        Sets protected attributes of early stopper.

        Parameters
        ----------
        name : str
            The name of the callback.
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        """
        super().__init__(name=name, **kwargs)

        self.best_measure = None
        self.counter = 0
        self.patience = patience
        self.tolerance = tolerance

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.LOW_PRIORITY.value

    @abstractmethod
    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
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


class MetricEarlyStopping(BaseEarlyStopping):

    def __init__(
            self,
            name: str,
            patience: int = 10,
            tolerance: float = 1e-4,
            **kwargs
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        name : str
            The name of the callback.
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        """
        super().__init__(name=name, patience=patience, tolerance=tolerance, **kwargs)

        self._tasks = []
        self._best_val_metric_scores = []

    @property
    def tasks(self) -> List[Task]:
        return self._tasks

    @tasks.setter
    def tasks(self, tasks: List[Task]):
        if not self._best_val_metric_scores:
            self._best_val_metric_scores = [
                np.inf if t.optimization_metric.direction == Direction.MINIMIZE.value else -np.inf for t in tasks
            ]

        self._tasks = tasks

    def on_epoch_end(self, trainer, **kwargs):
        """
        Checks if the training has to be stopped.

        Parameters
        ----------
        trainer : Trainer
            Trainer.
        kwargs : dict
        """
        self.tasks = trainer.state.tasks
        val_scores = trainer.state.valid_metrics

        new_scores_is_better = []
        for i, task, best_score in enumerate(zip(self._tasks, self._best_val_metric_scores)):
            val_score = val_scores[task.optimization_metric.name]

            if task.optimization_metric.direction == Direction.MINIMIZE.value:
                new_score_is_better = (best_score - val_score) > self.tolerance
            else:
                new_score_is_better = (val_score - best_score) > self.tolerance

            new_scores_is_better.append(new_score_is_better)

            if new_score_is_better:
                self._best_val_metric_scores[i] = val_score

        # if all the scores are worst than the best scores, we increment the counter
        if not any(new_scores_is_better):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                trainer.update_state_(stop_training_flag=True)
        else:
            self.counter = 0

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

        for score, task in zip(self._best_val_metric_scores, self._tasks):
            print(f"Task ({task.name}) (metric {task.optimization_metric.name}), Score :{round(score, 4)}")


class MultiTaskLossEarlyStopping(BaseEarlyStopping):

    def __init__(
            self,
            name: str,
            patience: int,
            tolerance: float,
            **kwargs
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        name : str
            The name of the callback.
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        """
        super().__init__(name=name, patience=patience, tolerance=tolerance, **kwargs)

        self._criterion = None
        self._is_better = lambda x, y: (y - x) > self.tolerance
        self._best_val_loss = np.inf

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
        print(f"Criterion {self._criterion.name}, Loss :{round(self._best_val_loss, 4)}")

    def on_epoch_end(self, trainer, **kwargs):
        """
        Checks if the training has to be stopped.

        Parameters
        ----------
        trainer : Trainer
            Trainer.
        kwargs : dict
        """
        val_loss = trainer.state.valid_losses[trainer.criterion.name]

        # if the score is worst than the best score we increment the counter
        if not self._is_better(val_loss, self._best_val_loss):
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                trainer.update_state_(stop_training_flag=True)

        # if the score is better than the best score saved we update the best model
        else:
            self._best_val_loss = val_loss
            self.counter = 0
