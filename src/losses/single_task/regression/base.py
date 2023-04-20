"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `RegressionLoss` class.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, nan, tensor, Tensor

from ..base import SingleTaskLoss, LossReduction
from ....tools.missing_targets import get_idx_of_nonmissing_regression_targets


class RegressionLoss(SingleTaskLoss, ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as regression criteria.
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
            Name of the Loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction)

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_regression_targets

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tensor:
        """
        Converts inputs to tensors than computes the Loss and applies rounding.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor or array with predicted labels.
        targets : Union[np.array, Tensor]
            (N,) tensor or array with ground truth

        Returns
        -------
        loss : Tensor
            Rounded loss score.
        """
        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        if len(nonmissing_targets_idx) == 0:
            return tensor(nan, device=pred.device)

        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self.perform_reduction(self._compute_loss(pred, targets))

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
            (N,) tensor or array containing predictions.
        targets : Union[np.array, Tensor]
            (N,) tensor or array containing ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
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
            (N,) tensor with predicted labels
        targets : Tensor
            (N,) tensor with ground truth

        Returns
        -------
        loss : Tensor
            Loss.
        """
        raise NotImplementedError
