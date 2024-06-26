"""
    @file:              base.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `BinaryClassificationMetric` class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, Tensor, zeros

from ..base import Direction, SingleTaskMetric, MetricReduction
from ....tools.missing_targets import get_idx_of_nonmissing_classification_targets


class BinaryClassificationMetric(SingleTaskMetric, ABC):
    """
    An Abstract class that represents the skeleton of callable classes to use as classification metrics
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            threshold: float = 0.5,
            weight: float = 0.5,
            n_digits: int = 7
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        threshold : float
            The threshold used to classify a sample in class 1.
        weight : float
            The weight attributed to class 1 (in [0, 1]).
        n_digits : int
            Number of digits kept.
        """
        if not (0 <= weight <= 1):
            raise ValueError("The weight parameter must be included in range [0, 1]")

        self._scaling_factor = None
        self._threshold = threshold
        self._weight = weight

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_classification_targets

        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    @property
    def scaling_factor(self) -> Optional[float]:
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, scaling_factor: float):
        self._scaling_factor = scaling_factor

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    @property
    def weight(self) -> float:
        return self._weight

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            thresh: Optional[float] = None
    ) -> float:
        """
        Converts inputs to tensors, applies softmax if shape is different than expected and then computes the metric
        and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth
        thresh : Optional[float]
            The threshold used to classify a sample in class 1.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        assert isinstance(self.scaling_factor, np.float64), f"Scaling factor must be set before computing the " \
                                                            f"{self.__class__.__name__}. Use the method " \
                                                            f"'update_scaling_factor' or directly set the " \
                                                            f"'scaling_factor attribute'."

        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        if len(nonmissing_targets_idx) == 0:
            return np.nan
        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if thresh is None:
            thresh = self._threshold

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.perform_reduction(self._compute_metric(pred, targets, thresh)), self.n_digits)

    def update_scaling_factor(
            self,
            y_train: Union[np.array, Tensor]
    ):
        """
        Computes the positive scaling factor that needs to be applied to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N_train, ) tensor or array with targets used for training.
        """
        y_train = y_train[self.get_idx_of_nonmissing_targets(y_train)]

        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        self._scaling_factor = (n0/n1)*(self.weight/(1-self.weight))

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

    def get_confusion_matrix(
            self,
            pred_proba: Tensor,
            targets: Tensor,
            thresh: Optional[float] = None
    ) -> Tensor:
        """
        Gets the confusion matrix.

        Parameters
        ----------
        pred_proba : Tensor
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.
        thresh : Optional[float]
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        confusion_matrix : Tensor
            (2,2) tensor
        """
        if thresh is None:
            thresh = self._threshold

        # We initialize an empty confusion matrix
        conf_matrix = zeros(2, 2)

        # We fill the confusion matrix
        pred_labels = (pred_proba >= thresh).long()
        for t, p in zip(targets, pred_labels):
            conf_matrix[t, p] += 1

        return conf_matrix

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Union[float, Tensor]:
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
        metric_score : Union[float, Tensor]
            Score as a float or a (N, 1) tensor.
        """
        raise NotImplementedError
