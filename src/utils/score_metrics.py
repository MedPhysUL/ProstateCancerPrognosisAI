"""
    @file:              score_metrics.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file is used to define the metrics used to measure models' performance.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, Tensor, zeros


class TaskType(Enum):
    """
    Custom enum for task types.
    """
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    def __iter__(self):
        return iter([self.REGRESSION, self.CLASSIFICATION])


class Direction(Enum):
    """
    Custom enum for optimization directions
    """
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    def __iter__(self):
        return iter([self.MAXIMIZE, self.MINIMIZE])


class Reduction(Enum):
    """
    Custom enum for metric reduction choices
    """
    MEAN = "mean"
    SUM = "sum"
    GEO_MEAN = "geometric_mean"


class Metric(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as optimization metrics.
    """

    def __init__(
            self,
            direction: Direction,
            name: str,
            task_type: TaskType,
            n_digits: int = 5
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        task_type : TaskType
            Whether we want to perform "regression" or "classification".
        n_digits : int
            Number of digits kept.
        """
        if direction not in Direction:
            raise ValueError("direction must be in {'maximize', 'minimize'}")

        if task_type not in TaskType:
            raise ValueError("task_type must be in {'regression', 'classification'}")

        # Protected attributes
        self._direction = direction
        self._name = name
        self._task_type = task_type
        self._n_digits = n_digits

    @abstractmethod
    def __call__(
            self,
            *args,
            **kwargs
    ) -> float:
        """
        Gets metric value.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        raise NotImplementedError

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_digits(self) -> int:
        return self._n_digits


class RegressionMetric(Metric):
    """
    An abstract class that represents the skeleton of callable classes to use as regression metrics.
    """

    def __init__(
            self,
            direction: Direction,
            name: str,
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, task_type=TaskType.REGRESSION, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> float:
        """
        Converts inputs to tensors than computes the metric and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.compute_metric(pred, targets), self.n_digits)

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts inputs to tensors.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N,) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).float()
        else:
            return pred, targets

    @abstractmethod
    def compute_metric(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> float:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted labels
        targets : Tensor
            (N,) tensor with ground truth

        Returns
        -------
        metric_score : float
            Score.
        """
        raise NotImplementedError


class BinaryClassificationMetric(Metric):
    """
    An Abstract class that represents the skeleton of callable classes to use as classification metrics
    """

    def __init__(
            self,
            direction: Direction,
            name: str,
            n_digits: int = 5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Direction
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, task_type=TaskType.CLASSIFICATION, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            thresh: float = 0.5
    ) -> float:
        """
        Converts inputs to tensors, applies softmax if shape is different than expected and than computes the metric
        and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth
        thresh : float
            The threshold used to classify a sample in class 1.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.compute_metric(pred, targets, thresh), self.n_digits)

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts predictions to float (since they are probabilities) and ground truth to long.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Union[np.array, Tensor]
            (N,) tensor with ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).long()
        else:
            return pred, targets

    @staticmethod
    def get_confusion_matrix(
            pred_proba: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Gets the confusion matrix.

        Parameters
        ----------
        pred_proba : Tensor
            (N,) tensor with with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.
        thresh : float
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        confusion_matrix : Tensor
            (2,2) tensor
        """
        # We initialize an empty confusion matrix
        conf_matrix = zeros(2, 2)

        # We fill the confusion matrix
        pred_labels = (pred_proba >= thresh).long()
        for t, p in zip(targets, pred_labels):
            conf_matrix[t, p] += 1

        return conf_matrix

    @abstractmethod
    def compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> float:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.
        thresh : float
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        metric_score : float
            Metric score.
        """
        raise NotImplementedError
