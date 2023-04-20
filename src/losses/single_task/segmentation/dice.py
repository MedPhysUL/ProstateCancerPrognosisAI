"""
    @file:              dice.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `DiceLoss` class.
"""

from typing import Optional, Union

from monai.losses import DiceLoss as _MonaiDiceLoss
from torch import Tensor

from .base import SegmentationLoss
from ..base import LossReduction


class DiceLoss(SegmentationLoss):
    """
    Callable class that computes the DICE loss.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        name : Optional[str]
            Name of the loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction)

        if self.reduction not in (LossReduction.MEAN.value, LossReduction.SUM.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Tensor:
        """
        Returns average Dice loss between two tensors.

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
        loss = _MonaiDiceLoss(sigmoid=True, reduction=LossReduction.NONE).to(device=pred.device)

        return loss(pred, targets)
