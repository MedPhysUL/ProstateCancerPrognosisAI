"""
    @file:              binary_accuracy.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `BinaryAccuracy` class.
"""

from typing import Optional, Union

from torch import Tensor

from .base import BinaryClassificationMetric
from ..base import Direction, MetricReduction


class BinaryAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes the binary accuracy.
    """

    def __init__(
            self,
            n_digits: int = 7,
            name: Optional[str] = None,
            reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        name : Optional[str]
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        """
        super().__init__(direction=Direction.MAXIMIZE, name=name, reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN, MetricReduction.SUM):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Returns the binary accuracy.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : Tensor
            Probability threshold that must be reach by a sample to be classified into class 1.

        Returns
        -------
        metric : Tensor
            (N, 1) tensor.
        """
        pred_labels = (pred >= thresh).float()

        return (pred_labels == targets).float()
