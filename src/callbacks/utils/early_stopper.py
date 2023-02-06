"""
    @file:              early_stopper.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `EarlyStopper` object, used within a `LearningAlgorithm`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np

from src.training.states import EpochState
from src.utils.metrics import Direction

if TYPE_CHECKING:
    from src.callbacks.learning_algorithm import LearningAlgorithm


class EarlyStopper(ABC):
    """
    Base class for early stopping.
    """

    def __init__(
            self,
            patience: int = 10,
            tolerance: float = 1e-4
    ) -> None:
        """
        Sets protected attributes of early stopper.

        Parameters
        ----------
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        """
        super().__init__()

        self.counter = 0
        self.learning_algorithm_name = None
        self.patience = patience
        self.tolerance = tolerance

    def on_fit_start(self, learning_algorithm: LearningAlgorithm):
        """
        Initializes early stopper on fit start.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        """
        self.learning_algorithm_name = learning_algorithm.name

    @abstractmethod
    def __call__(self, epoch_state: EpochState) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        raise NotImplementedError

    @abstractmethod
    def print_early_stopping_message(
            self,
            epoch_state: EpochState
    ) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.
        """
        raise NotImplementedError


class MetricsEarlyStopper(EarlyStopper):

    def __init__(
            self,
            patience: int = 10,
            tolerance: float = 1e-4
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        """
        super().__init__(patience=patience, tolerance=tolerance)

        self._best_val_metric_scores = []
        self._tasks = []

    def _initialize_best_val_metric_scores(self):
        """
        Initializes best validation metric scores depending on metrics' direction.
        """
        self._best_val_metric_scores = [
            np.inf if t.early_stopping_metric.direction == Direction.MINIMIZE.value
            else -np.inf for t in self._tasks
        ]

    def on_fit_start(self, learning_algorithm: LearningAlgorithm):
        """
        Sets learning algorithm and best validation metric scores.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        """
        super().on_fit_start(learning_algorithm)

        tasks = learning_algorithm.criterion.tasks
        assert all(task.early_stopping_metric is not None for task in tasks), (
            f"'MetricsEarlyStopper' requires that all tasks define the 'early_stopping_metric' attribute at instance "
            f"initialization."
        )

        self._tasks = tasks
        self._initialize_best_val_metric_scores()

    def __call__(self, epoch_state: EpochState) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        val_scores = epoch_state.valid.single_task_losses

        new_scores_is_better = []
        for i, task, best_score in enumerate(zip(self._tasks, self._best_val_metric_scores)):
            val_score = val_scores[task.name][task.early_stopping_metric.name]

            if task.early_stopping_metric.direction == Direction.MINIMIZE.value:
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
                return True
        else:
            self.counter = 0

        return False

    # TODO : Here, use logging instead of print.
    def print_early_stopping_message(
            self,
            epoch_state: EpochState
    ) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.
        """
        print(
            f"\n{self.learning_algorithm_name}: Early stopping occurred at epoch {epoch_state.idx} with best_epoch = "
            f"{epoch_state.idx - self.patience}"
        )

        for score, task in zip(self._best_val_metric_scores, self._tasks):
            print(f"\nTask ({task.name}) (metric {task.early_stopping_metric.name}), Score :{round(score, 4)}")


class MultiTaskLossEarlyStopper(EarlyStopper):

    def __init__(
            self,
            patience: int = 10,
            tolerance: float = 1e-4,
            include_regularization: Optional[bool] = True
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        include_regularization : Optional[bool]
            Whether to monitor the multi-task loss with or without regularization.
        """
        super().__init__(patience=patience, tolerance=tolerance)

        self.criterion_full_name = None
        self.include_regularization = include_regularization

        self._best_val_loss = np.inf

    def _set_criterion_full_name(self, learning_algorithm: LearningAlgorithm):
        """
        Initializes criterion full name.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            Learning algorithm.
        """
        basic_name = learning_algorithm.criterion.name

        if not learning_algorithm.regularization:
            suffix = EpochState.SUFFIX_WITHOUT_REGULARIZATION
        else:
            if self.include_regularization:
                suffix = EpochState.SUFFIX_WITH_REGULARIZATION
            else:
                suffix = EpochState.SUFFIX_WITHOUT_REGULARIZATION

        self.criterion_full_name = f"{basic_name}{suffix}"

    def on_fit_start(self, learning_algorithm: LearningAlgorithm):
        """
        Sets criterion tasks.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        """
        super().on_fit_start(learning_algorithm)
        self._set_criterion_full_name(learning_algorithm)

    def __call__(self, epoch_state: EpochState) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        val_loss = epoch_state.valid.multi_task_losses[self.learning_algorithm_name][self.criterion_full_name]

        # if the score is worst than the best score we increment the counter
        if not (self._best_val_loss - val_loss) > self.tolerance:
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                return True

        # if the score is better than the best score saved we update the best model
        else:
            self._best_val_loss = val_loss
            self.counter = 0

        return False

    # TODO : Here, use logging instead of print.
    def print_early_stopping_message(
            self,
            epoch_state: EpochState
    ) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        epoch_state : EpochState
            The current epoch state.
        """
        print(
            f"\n{self.learning_algorithm_name}: Early stopping occurred at epoch {epoch_state.idx} with best_epoch = "
            f"{epoch_state.idx - self.patience}. \nCriterion {self.criterion_full_name}, "
            f"Loss :{round(self._best_val_loss, 4)}"
        )
