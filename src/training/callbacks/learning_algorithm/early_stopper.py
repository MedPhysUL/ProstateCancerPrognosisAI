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
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from torch import load, save, tensor

from ....losses.multi_task.base import MultiTaskLoss
from ....metrics.single_task.base import Direction

if TYPE_CHECKING:
    from ..learning_algorithm import LearningAlgorithm
    from ....models.torch.base import TorchModel


class EarlyStopper(ABC):
    """
    Base class for early stopping.
    """

    BEST_MODEL_NAME = "best_model"

    # Identical suffix as EpochState
    SUFFIX_WITH_REGULARIZATION = "('regularization'=True)"
    SUFFIX_WITHOUT_REGULARIZATION = "('regularization'=False)"

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
    """
    Early stopper based on metrics. Stops when the validation metric does not improve for a given number of epochs.
    """

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
    """
    Early stopper based on multi-task loss. Stops when the validation multi-task loss does not improve for a given
    number of epochs. The multi-task loss can be computed with or without regularization.
    """

    EARLY_STOPPING_PREFIX = "EarlyStopping"

    def __init__(
            self,
            criterion: Optional[MultiTaskLoss] = None,
            include_regularization: Optional[bool] = True,
            patience: int = 10,
            tolerance: float = 1e-3
    ) -> None:
        """
        Sets protected attributes of early stopper and defines comparison methods according to the given tasks.

        Parameters
        ----------
        criterion : Optional[MultiTaskLoss]
            Multi-task loss. If None, the criterion of the learning algorithm will be used.
        patience : int
            Number of consecutive epochs without improvement allowed.
        tolerance : float
            Permissible difference between measures.
        include_regularization : Optional[bool]
            Whether to monitor the multi-task loss with or without regularization.
        """
        super().__init__(patience=patience, tolerance=tolerance)

        self.best_val_loss = None
        self.criterion = criterion
        self.original_criterion_name = None
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
        if not learning_algorithm.regularizer:
            suffix = self.SUFFIX_WITHOUT_REGULARIZATION
        else:
            if self.include_regularization:
                suffix = self.SUFFIX_WITH_REGULARIZATION
            else:
                suffix = self.SUFFIX_WITHOUT_REGULARIZATION

        self.criterion_full_name = f"{self.original_criterion_name}{suffix}"

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
        self.original_criterion_name = learning_algorithm.criterion.name
        if self.criterion is None:
            self._set_criterion_full_name(learning_algorithm)

    def _get_train_and_valid_losses(self, trainer) -> Tuple[float, float]:
        """
        Gets the train and validation losses.

        Parameters
        ----------
        trainer : Trainer
            Trainer

        Returns
        -------
        train_loss, valid_loss : Tuple[float, float]
            Train and validation losses.
        """
        train_dict, valid_dict = {}, {}
        for task in self.criterion.tasks:
            train_dict[task.name] = tensor(trainer.epoch_state.train.single_task_losses[task.name][task.criterion.name])
            valid_dict[task.name] = tensor(trainer.epoch_state.valid.single_task_losses[task.name][task.criterion.name])

        return self.criterion(train_dict).item(), self.criterion(valid_dict).item()

    def _set_multi_task_losses(self, trainer, train_loss: float, valid_loss: float, regularized: bool):
        """
        Sets the multi-task losses.

        Parameters
        ----------
        trainer : Trainer
            Trainer
        train_loss : float
            Train loss.
        valid_loss : float
            Validation loss.
        regularized : bool
            Whether to include regularization or not.
        """
        if regularized:
            loss_name = f"{self.EARLY_STOPPING_PREFIX}_{self.criterion.name}{self.SUFFIX_WITH_REGULARIZATION}"
        else:
            loss_name = f"{self.EARLY_STOPPING_PREFIX}_{self.criterion.name}{self.SUFFIX_WITHOUT_REGULARIZATION}"

        train_multi_task_losses = trainer.epoch_state.train.multi_task_losses[self.learning_algorithm_name]
        valid_multi_task_losses = trainer.epoch_state.valid.multi_task_losses[self.learning_algorithm_name]

        train_multi_task_losses[loss_name], valid_multi_task_losses[loss_name] = train_loss, valid_loss

    def _get_regularization_module(self, trainer, train: bool) -> float:
        """
        Gets the regularization module.

        Parameters
        ----------
        trainer : Trainer
            Trainer
        train : bool
            Whether to get the regularization module of the train or validation epoch.

        Returns
        -------
        regularization_module : float
            Regularization module.
        """
        if train:
            multi_task_losses = trainer.epoch_state.train.multi_task_losses[self.learning_algorithm_name]
        else:
            multi_task_losses = trainer.epoch_state.valid.multi_task_losses[self.learning_algorithm_name]

        loss_with_reg = multi_task_losses[f"{self.original_criterion_name}{self.SUFFIX_WITH_REGULARIZATION}"]
        loss_without_reg = multi_task_losses[f"{self.original_criterion_name}{self.SUFFIX_WITHOUT_REGULARIZATION}"]

        return loss_with_reg - loss_without_reg

    def _compute_validation_loss(self, trainer) -> float:
        """
        Computes the early stopping validation loss.

        Parameters
        ----------
        trainer : Trainer
            Trainer

        Returns
        -------
        early_stopping_loss : float
            Early stopping validation loss.
        """
        if self.criterion:
            train_loss, valid_loss = self._get_train_and_valid_losses(trainer)
            self._set_multi_task_losses(trainer, train_loss, valid_loss, regularized=False)

            if self.include_regularization:
                train_regularization = self._get_regularization_module(trainer, train=True)
                valid_regularization = self._get_regularization_module(trainer, train=False)

                train_loss, valid_loss = train_loss + train_regularization, valid_loss + valid_regularization
                self._set_multi_task_losses(trainer, train_loss, valid_loss, regularized=True)

                return valid_loss
            else:
                return valid_loss
        else:
            return trainer.epoch_state.valid.multi_task_losses[self.learning_algorithm_name][self.criterion_full_name]

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
        val_loss = self._compute_validation_loss(trainer)

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
