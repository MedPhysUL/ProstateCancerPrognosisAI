"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the abstract `SurvivalAnalysisMetric` class.
"""

from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor

from ..base import Direction, MetricReduction
from ..regression import RegressionMetric
from ....tools.missing_targets import get_idx_of_nonmissing_survival_analysis_targets


# TODO

class SurvivalAnalysisMetric(RegressionMetric, ABC):
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

    @abstractmethod
    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Union[float, Tensor]:
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
        metric_score : Union[float, Tensor]
            Score as a float or a (N, 1) tensor.
        """
        raise NotImplementedError
