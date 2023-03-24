"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the abstract `SurvivalAnalysisMetric` class.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, Tensor

from ..base import Direction, MetricReduction, SingleTaskMetric
from ....tools.missing_targets import get_idx_of_nonmissing_survival_analysis_targets


class SurvivalAnalysisMetric(SingleTaskMetric, ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as survival analysis metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
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
        n_digits : int
            Number of digits kept.
        """
        super().__init__(direction=direction, name=name, reduction=reduction, n_digits=n_digits)

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_survival_analysis_targets

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
        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        if len(nonmissing_targets_idx) == 0:
            return np.nan

        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return round(self.perform_reduction(self._compute_metric(pred, targets[:, 0], targets[:, 1])), self.n_digits)

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
    def _compute_metric(
            self,
            pred: Tensor,
            event_indicator: Tensor,
            event_time: Tensor
    ) -> Union[float, Tensor]:
        """
        Computes the metric score.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted labels.
        event_indicator : Tensor
            (N,) tensor with event indicators.
        event_time : Tensor
            (N,) tensor with event times.

        Returns
        -------
        metric_score : Union[float, Tensor]
            Score as a float or a (N, 1) tensor.
        """
        raise NotImplementedError

