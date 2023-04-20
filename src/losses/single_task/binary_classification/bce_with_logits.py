"""
    @file:              bce_with_logits.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `BCEWithLogitsLoss` class.
"""

from typing import Optional, Union

from torch import nn, Tensor, where

from .base import BinaryClassificationLoss
from ..base import LossReduction


class BCEWithLogitsLoss(BinaryClassificationLoss):
    """
    Callable class that computes binary cross entropy.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            weight: float = 0.5,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        name : Optional[str]
            Name of the loss.
        weight : float
            The weight attributed to class 1 (in [0, 1]).
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction, weight=weight)

        if self.reduction not in (LossReduction.MEAN, LossReduction.SUM):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Tensor:
        """
        Returns the Binary Cross Entropy between the target and the input probabilities.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        loss = nn.BCEWithLogitsLoss(
            weight=where(targets == 1, self.scaling_factor, 1),
            reduction=LossReduction.NONE
        ).to(device=pred.device)

        return loss(pred, targets.float())
