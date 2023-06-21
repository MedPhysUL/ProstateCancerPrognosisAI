"""
    @file:              sensitivity.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the `Sensitivity` class.
"""

from typing import Optional

from torch import Tensor

from .base import BinaryClassificationMetric
from ..base import Direction, MetricReduction


class Sensitivity(BinaryClassificationMetric):
    """
    Callable class that computes the sensitivity -> TP/(TP + FN).
    """

    def __init__(
            self,
            n_digits: int = 7,
            name: Optional[str] = None
    ):
        """
        Sets protected attributes using parent's constructor. Note that the direction is set to NONE instead of MAXIMUM
        since maximizing this metric will lead to a threshold of 0.

        Parameters
        ----------
        n_digits : int
            Number of digits kept for the score.
        name : Optional[str]
            Name of the metric.
        """
        super().__init__(direction=Direction.NONE, name=name, reduction=MetricReduction.NONE, n_digits=n_digits)

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Returns the sensitivity, i.e. TP/(TP + FN).

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : float
            Probability threshold that must be reach by a sample to be classified into class 1 (Not used here)

        Returns
        -------
        metric : Tensor
            (N, 1) tensor.
        """
        # We first extract the confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We compute TP/(TP + FN)
        return conf_mat[1, 1]/(conf_mat[1, 1] + conf_mat[1, 0])
