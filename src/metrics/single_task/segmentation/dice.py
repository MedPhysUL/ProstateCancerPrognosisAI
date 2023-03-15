"""
    @file:              dice.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `DiceMetric` class.
"""

from typing import Optional, Union

from monai.metrics import DiceMetric as _MonaiDiceMetric
from torch import Tensor

from .base import SegmentationMetric
from ..base import Direction, MetricReduction


class DiceMetric(SegmentationMetric):
    """
    Callable class that computes the DICE.
    """

    def __init__(
            self,
            n_digits: int = 7,
            name: Optional[str] = None,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        name : Optional[str]
            Name of the metric.
        reduction :  Union[MetricReduction, str]
            Reduction method to use.
        """
        super().__init__(direction=Direction.MAXIMIZE, name=name, reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN, MetricReduction.SUM):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Tensor:
        """
        Returns the average of the DICE score.

        Parameters
        ----------
        pred : Tensor
            (B, C, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (B, C, X, Y, Z) tensor with ground truth

        Returns
        -------
        metric : Tensor
            (N, 1) tensor.
        """
        metric = _MonaiDiceMetric(reduction="mean")

        return metric(y_pred=pred, y=targets)
