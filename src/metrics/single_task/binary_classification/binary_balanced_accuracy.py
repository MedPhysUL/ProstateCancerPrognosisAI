"""
    @file:              binary_balanced_accuracy.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `BinaryBalancedAccuracy` class.
"""

from typing import Optional, Union

from torch import Tensor

from .base import BinaryClassificationMetric
from ..base import Direction, MetricReduction


class BinaryBalancedAccuracy(BinaryClassificationMetric):
    """
    Callable class that computes balanced accuracy using confusion matrix.
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
            "Mean" for (TPR + TNR)/2 or "GeometricMean" for sqrt(TPR*TNR)
        """
        super().__init__(direction=Direction.MAXIMIZE, name=name, reduction=reduction, n_digits=n_digits)

        if self.reduction not in (MetricReduction.MEAN, MetricReduction.GEOMETRIC_MEAN):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are "
                             f"['mean', 'geometric_mean'].")

    def _compute_metric(
            self,
            pred: Tensor,
            targets: Tensor,
            thresh: float
    ) -> Tensor:
        """
        Returns the balanced accuracy, i.e (TPR + TNR)/2.

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
        # We get confusion matrix
        conf_mat = self.get_confusion_matrix(pred, targets, thresh)

        # We get TNR and TPR
        correct_rates = conf_mat.diag() / conf_mat.sum(dim=1)

        return correct_rates
