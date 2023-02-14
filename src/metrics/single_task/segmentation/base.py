"""
    @file:              base.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `SegmentationMetric` class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, Tensor

from ..base import Direction, SingleTaskMetric, MetricReduction


class SegmentationMetric(SingleTaskMetric, ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as segmentation metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            n_digits: int = 5
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
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            reduction: Optional[Union[MetricReduction, str]] = None
    ) -> float:
        """
        Converts inputs to tensors then computes the metric and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with ground truth
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use. If None, we use the default reduction, i.e. self.reduction.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.perform_reduction(self._compute_metric(pred, targets), reduction), self.n_digits)

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
            (N, X, Y, Z) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N, X, Y, Z) tensor, (N, X, Y, Z) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).float()
        else:
            return pred, targets

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> float:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (B, C, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (B, C, X, Y, Z) tensor with ground truth

        Returns
        -------
        metric_score : float
            Score as a float.
        """
        raise NotImplementedError
