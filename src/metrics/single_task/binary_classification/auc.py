"""
    @file:              auc.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `AUC` class.
"""

from typing import Optional

from sklearn.metrics import roc_auc_score
from torch import Tensor

from .base import BinaryClassificationMetric
from ..base import Direction, MetricReduction


class AUC(BinaryClassificationMetric):
    """
    Callable class that computes the AUC for ROC curve.
    """

    def __init__(
            self,
            n_digits: int = 5,
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
            targets: Tensor,
            thresh: float
    ) -> float:
        """
        Returns the AUC for ROC curve.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        thresh : Tensor
            Probability threshold that must be reach by a sample to be classified into class 1 (Not used here)

        Returns
        -------
        metric : float
            Score.
        """
        return roc_auc_score(targets.numpy(), pred.numpy())
