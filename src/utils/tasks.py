"""
    @file:              tasks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the different possible tasks.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from torch import isnan, Tensor, where

from src.utils.score_metrics import BinaryClassificationMetric, Metric, RegressionMetric, SegmentationMetric


class TaskType(Enum):
    """
    Custom enum for task types.
    """
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"

    def __iter__(self):
        return iter([self.REGRESSION, self.CLASSIFICATION, self.SEGMENTATION])


class Task(ABC):
    """
    An abstract class representing a task.
    """

    def __init__(
            self,
            metric: Metric
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : Metric
            A score metric.
        """
        self._metric = metric

    @property
    def metric(self) -> Metric:
        return self._metric


class ImagingTask(Task, ABC):
    """
    An abstract class representing a task to do on imaging data.
    """

    def __init__(
            self,
            metric: Metric
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : Metric
            A score metric.
        """
        super().__init__(metric=metric)


class TableTask(Task, ABC):
    """
    An abstract class representing a task to do on table data.
    """

    def __init__(
            self,
            metric: Metric,
            target_col: str
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : Metric
            A score metric.
        target_col : str
            Name of the column containing the targets associated to this task.
        """
        self._target_col = target_col

        super().__init__(metric=metric)

    @property
    def target_col(self) -> str:
        return self._target_col

    @abstractmethod
    def get_idx_of_nonmissing_targets(
            self,
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Indices.
        """
        raise NotImplementedError


class ClassificationTask(TableTask):
    """
    A class used to define a Classification task.
    """

    def __init__(
            self,
            metric: BinaryClassificationMetric,
            target_col: str,
            weight: float = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : BinaryClassificationMetric
            A score metric.
        target_col : str
            Name of the column containing the targets associated to this task.
        weight : Optional[float]
            The weight attributed to class 1 (in [0, 1]).
        """
        super().__init__(metric=metric, target_col=target_col)
        if weight is not None:
            if not (0 < weight < 1):
                raise ValueError("The weight parameter must be included in range [0, 1]")

        self._weight = weight

    @property
    def weight(self) -> Optional[float]:
        return self._weight

    def get_idx_of_nonmissing_targets(
            self,
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(y >= 0)
        else:
            idx = np.where(y >= 0)

        return idx[0].tolist()

    def get_scaling_factor(
            self,
            y_train: Union[np.array, Tensor]
    ) -> float:
        """
        Computes the scaling factor that needs to be apply to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N_train, ) tensor or array with targets used for training (Ex : y_train = ds.y[ds.train_mask, task_idx]).

        Returns
        -------
        scaling_factor : float]
            Positive scaling factors.
        """
        y_train = y_train[self.get_idx_of_nonmissing_targets(y_train)]

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        return (n0/n1)*(self.weight/(1-self.weight))


class RegressionTask(TableTask):
    """
    A class used to define a Regression task.
    """

    def __init__(
            self,
            metric: RegressionMetric,
            target_col: str
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : RegressionMetric
            A score metric.
        target_col : str
            Name of the column containing the targets associated to this task.
        """
        super().__init__(metric=metric, target_col=target_col)

    def get_idx_of_nonmissing_targets(
            self,
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(~isnan(y))
        else:
            idx = np.where(~np.isnan(y))

        return idx[0].tolist()


class SegmentationTask(ImagingTask):
    """
    A class used to define a Segmentation task.
    """

    def __init__(
            self,
            metric: SegmentationMetric,
            organ: str
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        metric : SegmentationMetric
            A score metric.
        organ : str
            Segmented organ.
        """
        self._organ = organ
        super().__init__(metric=metric)

    @property
    def organ(self) -> str:
        return self._organ

    @property
    def task_type(self) -> TaskType:
        return TaskType.SEGMENTATION
