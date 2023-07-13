"""
    @file:              cumulative_dynamic_auc.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file is used to define the abstract `CumulativeDynamicAUC` class.
"""

from typing import Optional

import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from torch import Tensor

from .base import SurvivalAnalysisMetric
from ..base import Direction, MetricReduction


class CumulativeDynamicAUC(SurvivalAnalysisMetric):
    """
    Callable class that computes the cumulative/dynamic AUC for right-censored time-to-event data.
    """

    def __init__(
            self,
            time: float = 60.0,
            n_digits: int = 7,
            name: Optional[str] = None
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        time : float
            Time at which the AUC is computed. Defaults to 60.0.
        n_digits : int
            Number of digits kept for the score.
        name : Optional[str]
            Name of the metric.
        """
        super().__init__(direction=Direction.MAXIMIZE, name=name, reduction=MetricReduction.NONE, n_digits=n_digits)
        self._time = time

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
        event_indicator = event_indicator.numpy().astype(bool)
        event_time = event_time.numpy()

        if not event_indicator.any():
            return np.nan

        cda = cumulative_dynamic_auc(
            survival_train=self._get_structured_array(self._training_event_indicator, self._training_event_time),
            survival_test=self._get_structured_array(event_indicator, event_time),
            estimate=pred.numpy(),
            times=self._time
        )

        return cda[1]
