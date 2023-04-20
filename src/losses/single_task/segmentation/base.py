"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `SegmentationLoss` class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, Tensor

from ..base import SingleTaskLoss, LossReduction


class SegmentationLoss(SingleTaskLoss, ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as segmentation criteria.
    """

    def __init__(
            self,
            name: str,
            reduction: Union[LossReduction, str],
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        name : str
            Name of the loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction)

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor],
            reduction: Optional[Union[LossReduction, str]] = None
    ) -> Tensor:
        """
        Converts inputs to tensors than computes the loss value and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array with ground truth
        reduction : Optional[Union[LossReduction, str]]
            Reduction method to use. If None, we use the default reduction, i.e. self.reduction.

        Returns
        -------
        loss : Tensor
            Rounded loss value.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self.perform_reduction(self._compute_loss(pred, targets), reduction)

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts inputs to tensors.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N, X, Y, Z) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N, X, Y, Z) tensor, (N, X, Y, Z) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).float()
        else:
            return pred, targets

    @abstractmethod
    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        pred : Tensor
            (N, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (N, X, Y, Z) tensor with ground truth

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        raise NotImplementedError
