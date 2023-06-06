"""
    @file:              early_stopper.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `EarlyStopper` object, used within a `LearningAlgorithm`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from os import path
from typing import Optional, TYPE_CHECKING

import numpy as np
from torch import load, save

from ....metrics.single_task.base import Direction

if TYPE_CHECKING:
    from ..learning_algorithm import LearningAlgorithm
    from ....models.torch.base import TorchModel


class EarlyStopper(ABC):
    """
    Base class for early stopping.
    """

    BEST_MODEL_NAME = "best_model"

    def __init__(
            self,
            patience: int = 10,
            tolerance: float = 1e-3
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

        self.counter = None
        self.learning_algorithm_name = None
        self.path_to_best_model = None
        self.patience = patience
        self.tolerance = tolerance

    @abstractmethod
    def __call__(self, trainer) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        trainer : Trainer
            The current trainer.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        raise NotImplementedError

    def on_fit_start(self, learning_algorithm: LearningAlgorithm, trainer):
        """
        Initializes early stopper on fit start.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        trainer : Trainer
            Trainer
        """
        assert trainer.training_state.valid_dataloader, (
            "Early stopping is not available if 'validation_set_size' == 0. Update the masks of the dataset to add "
            "samples in the validation set."
        )
        self.learning_algorithm_name = learning_algorithm.name
        self.path_to_best_model = path.join(
            trainer.training_state.path_to_temporary_folder,
            f"{self.learning_algorithm_name}_{self.BEST_MODEL_NAME}.pth"
        )
        self.counter = 0

    def load_best_model(self, model):
        """
        Loads best model.

        Parameters
        ----------
        model : TorchModel
            Model.
        """
        model.load_state_dict(load(self.path_to_best_model))

    def set_best_epoch(self, trainer):
        """
        Sets best epoch.

        Parameters
        ----------
        trainer : Trainer
            trainer
        """
        trainer.training_state.best_epoch = trainer.epoch_state.idx - self.patience

    def _exec_early_stopping(self, trainer):
        """
        Executes early stopping.

        Parameters
        ----------
        trainer : Trainer
            Trainer.
        """
        self.load_best_model(trainer.model)
        self.set_best_epoch(trainer)

        if trainer.verbose:
            self._print_early_stopping_message(trainer)

    @staticmethod
    def _print_early_stopping_message(trainer) -> None:
        """
        Prints a message when early stopping occurs.

        Parameters
        ----------
        trainer : Trainer
            The current trainer.
        """
        # TODO : Here, use logging instead of print.
        print(
            f"\nEarlyStopper: Early stopping occurred at epoch {trainer.epoch_state.idx} with best_epoch = "
            f"{trainer.training_state.best_epoch}."
        )


class MetricsEarlyStopper(EarlyStopper):

    def __init__(
            self,
            patience: int = 10,
            tolerance: float = 1e-3
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

        self._best_val_metric_scores = None
        self._tasks = None

    def _initialize_best_val_metric_scores(self):
        """
        Initializes best validation metric scores depending on metrics' direction.
        """
        self._best_val_metric_scores = [
            np.inf if t.early_stopping_metric.direction == Direction.MINIMIZE
            else -np.inf for t in self._tasks
        ]

    def on_fit_start(self, learning_algorithm: LearningAlgorithm, trainer):
        """
        Sets learning algorithm and best validation metric scores.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        trainer : Trainer
            Trainer
        """
        super().on_fit_start(learning_algorithm, trainer)

        tasks = learning_algorithm.criterion.tasks
        assert all(task.early_stopping_metric is not None for task in tasks), (
            f"'MetricsEarlyStopper' requires that all tasks define the 'early_stopping_metric' attribute at instance "
            f"initialization."
        )

        self._tasks = tasks
        self._initialize_best_val_metric_scores()

    def __call__(self, trainer) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        trainer : Trainer
            The current trainer.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        val_scores = trainer.epoch_state.valid.single_task_metrics

        new_scores_is_better = []
        for i, (task, best_score) in enumerate(zip(self._tasks, self._best_val_metric_scores)):
            val_score = val_scores[task.name][task.early_stopping_metric.name]

            if task.early_stopping_metric.direction == Direction.MINIMIZE:
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
                self._exec_early_stopping(trainer)
                return True
        else:
            save(trainer.model.state_dict(), self.path_to_best_model)
            self.counter = 0

        return False


class MultiTaskLossEarlyStopper(EarlyStopper):

    # Identical suffix as EpochState
    SUFFIX_WITH_REGULARIZATION = "('regularization'=True)"
    SUFFIX_WITHOUT_REGULARIZATION = "('regularization'=False)"

    def __init__(
            self,
            include_regularization: Optional[bool] = True,
            patience: int = 10,
            tolerance: float = 1e-3
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

        self.best_val_loss = None
        self.criterion_full_name = None
        self.include_regularization = include_regularization

    def _set_criterion_full_name(self, learning_algorithm: LearningAlgorithm):
        """
        Initializes criterion full name.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            Learning algorithm.
        """
        basic_name = learning_algorithm.criterion.name

        if not learning_algorithm.regularizer:
            suffix = self.SUFFIX_WITHOUT_REGULARIZATION
        else:
            if self.include_regularization:
                suffix = self.SUFFIX_WITH_REGULARIZATION
            else:
                suffix = self.SUFFIX_WITHOUT_REGULARIZATION

        self.criterion_full_name = f"{basic_name}{suffix}"

    def on_fit_start(self, learning_algorithm: LearningAlgorithm, trainer):
        """
        Sets criterion tasks.

        Parameters
        ----------
        learning_algorithm : LearningAlgorithm
            The learning algorithm.
        trainer : Trainer
            Trainer
        """
        super().on_fit_start(learning_algorithm, trainer)
        self.best_val_loss = np.inf
        self._set_criterion_full_name(learning_algorithm)

    def __call__(self, trainer) -> bool:
        """
        Called when an epoch ends. Returns whether to early stop.

        Parameters
        ----------
        trainer : Trainer
            The current trainer.

        Returns
        -------
        early_stop : bool
            Whether to early stop.
        """
        val_loss = trainer.epoch_state.valid.multi_task_losses[self.learning_algorithm_name][self.criterion_full_name]

        # if the score is worst than the best score we increment the counter
        if not (self.best_val_loss - val_loss) > self.tolerance:
            self.counter += 1

            # if the counter reach the patience we early stop
            if self.counter >= self.patience:
                self._exec_early_stopping(trainer)
                return True
        # if the score is better than the best score saved we update the best model
        else:
            save(trainer.model.state_dict(), self.path_to_best_model)
            self.best_val_loss = val_loss
            self.counter = 0

        return False
