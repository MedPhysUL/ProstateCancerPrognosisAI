"""
    @file:              concordance_index_censored.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the abstract `ConcordanceIndexCensored` class.
"""

from typing import Optional

from sksurv.metrics import concordance_index_censored
from torch import Tensor

from .base import SurvivalAnalysisMetric
from ..base import Direction, MetricReduction


class ConcordanceIndexCensored(SurvivalAnalysisMetric):
    """
    Callable class that computes the censored concordance index.
    """

    def __init__(
            self,
            n_digits: int = 7,
            name: Optional[str] = None
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        name : Optional[str]
            Name of the metric.
        """
        super().__init__(direction=Direction.MAXIMIZE, name=name, reduction=MetricReduction.NONE, n_digits=n_digits)

    def _compute_metric(
            self,
            pred: Tensor,
            event_indicator: Tensor,
            event_time: Tensor
    ) -> float:
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
        metric : float
            Score.
        """
        c_index = concordance_index_censored(
            event_indicator=event_indicator.numpy().astype(bool),
            event_time=event_time.numpy(),
            estimate=pred.numpy()
        )
        return c_index[0]
