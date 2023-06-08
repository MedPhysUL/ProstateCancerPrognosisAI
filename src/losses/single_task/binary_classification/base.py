"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `BinaryClassificationLoss` class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from torch import from_numpy, is_tensor, nan, tensor, Tensor

from ..base import SingleTaskLoss, LossReduction
from ....tools.missing_targets import get_idx_of_nonmissing_classification_targets


class BinaryClassificationLoss(SingleTaskLoss, ABC):
    """
    An Abstract class that represents the skeleton of callable classes to use as classification criteria.
    """

    def __init__(
            self,
            name: str,
            reduction: Union[LossReduction, str],
            weight: float = 0.5
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        name : str
            Name of the loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        weight : float
            The weight attributed to class 1 (in [0, 1]).
        """
        if not (0 <= weight <= 1):
            raise ValueError("The weight parameter must be included in range [0, 1]")

        self._weight = weight
        self._scaling_factor = None

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_classification_targets

        super().__init__(name=name, reduction=reduction)

    @property
    def scaling_factor(self) -> Optional[float]:
        return self._scaling_factor

    @scaling_factor.setter
    def scaling_factor(self, scaling_factor: float):
        self._scaling_factor = scaling_factor

    @property
    def weight(self) -> float:
        return self._weight

    def __call__(
            self,
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tensor:
        """
        Converts inputs to tensors and than computes the loss and applies rounding.

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
        assert isinstance(self.scaling_factor, np.float64), f"Scaling factor must be set before computing the" \
                                                            f" {self.__class__.__name__}. Use the method " \
                                                            f"'update_scaling_factor' or directly set the " \
                                                            f"'scaling_factor attribute'."

        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        if len(nonmissing_targets_idx) == 0:
            return tensor(nan, device=pred.device)

        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self.perform_reduction(self._compute_loss(pred, targets))

    def update_scaling_factor(
            self,
            y_train: Union[np.array, Tensor]
    ):
        """
        Computes the positive scaling factor that needs to be apply to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N_train, ) tensor or array with targets used for training.
        """
        y_train = y_train[self.get_idx_of_nonmissing_targets(y_train)]

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        self._scaling_factor = (n0/n1)*(self.weight/(1-self.weight))

    @staticmethod
    def convert_to_tensors(
            pred: Union[np.array, Tensor],
            targets: Union[np.array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Converts predictions to float (since they are probabilities) and ground truth to long.

        Parameters
        ----------
        pred : Union[np.array, Tensor]
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Union[np.array, Tensor]
            (N,) tensor with ground truth.

        Returns
        -------
        pred, targets : Tuple[Tensor, Tensor]
            (N,) tensor, (N,) tensor
        """
        if not is_tensor(pred):
            return from_numpy(pred).float(), from_numpy(targets).long()
        else:
            return pred, targets

    @abstractmethod
    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor,
    ) -> Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1.
        targets : Tensor
            (N,) tensor with ground truth.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        raise NotImplementedError
