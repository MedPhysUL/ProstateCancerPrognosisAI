"""
    @file:              losses.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define the losses used to measure models' performance.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from monai.losses import DiceLoss
import numpy as np
from torch import from_numpy, isnan, is_tensor, mean, nn, sum, tensor, Tensor, where

from src.utils.reductions import LossReduction


class Loss(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as optimization criteria.
    """

    def __init__(
            self,
            name: str,
            reduction: Union[LossReduction, str],
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : str
            Name of the Loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        # Protected attributes
        self._name = name
        self._reduction = LossReduction(reduction).value

    @abstractmethod
    def __call__(
            self,
            *args,
            **kwargs
    ) -> Tensor:
        """
        Gets loss value.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    def perform_reduction(
            self,
            x: Tensor,
            reduction: Optional[Union[LossReduction, str]] = None
    ) -> Tensor:
        """
        Gets loss value.

        Parameters
        ----------
        x : Tensor
            (N, 1) tensor.
        reduction : Optional[Union[LossReduction, str]]
            Reduction method to use. If None, we use self.reduction.

        Returns
        -------
        loss : Tensor
            (1, 1) tensor.
        """
        if reduction is None:
            reduction = self.reduction
        else:
            reduction = LossReduction(reduction).value

        if reduction == LossReduction.NONE.value:
            return x
        elif reduction == LossReduction.MEAN.value:
            return mean(x)
        elif reduction == LossReduction.SUM.value:
            return sum(x)

    @property
    def reduction(self) -> str:
        return self._reduction


class RegressionLoss(Loss):
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
        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self.perform_reduction(self._compute_loss(pred, targets))

    @staticmethod
    def get_idx_of_nonmissing_targets(
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(~isnan(y))
        else:
            idx = np.where(~np.isnan(y))

        return idx[0].tolist()

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


class BinaryClassificationLoss(Loss):
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
        if not (0 < weight < 1):
            raise ValueError("The weight parameter must be included in range [0, 1]")

        self._weight = weight
        self._scaling_factor = None

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
        nonmissing_targets_idx = self.get_idx_of_nonmissing_targets(targets)
        targets, pred = targets[nonmissing_targets_idx], pred[nonmissing_targets_idx]

        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self.perform_reduction(self._compute_loss(pred, targets))

    @staticmethod
    def get_idx_of_nonmissing_targets(
            y: Union[Tensor, np.array]
    ) -> List[int]:
        """
        Gets the idx of the nonmissing targets in the given array or tensor.

        Parameters
        ----------
        y : Union[Tensor, np.array]
            (N,) tensor or array with targets.

        Returns
        -------
        idx : List[int]
            Index.
        """
        if isinstance(y, Tensor):
            idx = where(y >= 0)
        else:
            idx = np.where(y >= 0)

        return idx[0].tolist()

    def get_scaling_factor(
            self,
            y_train: Union[np.array, Tensor]
    ) -> float:
        """
        Computes the scaling factor that needs to be apply to the weight of samples in the class 1.

        We need to find alpha that satisfies :
            (alpha*n1)/n0 = w/(1-w)
        Which gives the solution:
            alpha = w*n0/(1-w)*n1

        Parameters
        ----------
        y_train : Union[Tensor, np.array]
            (N_train, ) tensor or array with targets used for training.

        Returns
        -------
        scaling_factor : float
            Positive scaling factors.
        """
        y_train = y_train[self.get_idx_of_nonmissing_targets(y_train)]

        # Otherwise we return samples' weights in the appropriate format
        n1 = y_train.sum()              # number of samples with label 1
        n0 = y_train.shape[0] - n1      # number of samples with label 0

        return (n0/n1)*(self.weight/(1-self.weight))

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


class SegmentationLoss(Loss):
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
            Reduction method to use. If None, we use self.reduction.

        Returns
        -------
        loss : Tensor
            Rounded loss value.
        """
        if not is_tensor(pred):
            pred, targets = self.convert_to_tensors(pred, targets)

        return self._compute_loss(pred, targets, reduction)

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
            targets: Tensor,
            reduction: Optional[Union[LossReduction, str]]
    ) -> Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        pred : Tensor
            (N, X, Y, Z) tensor with predicted labels
        targets : Tensor
            (N, X, Y, Z) tensor with ground truth
        reduction : Optional[Union[LossReduction, str]]
            Reduction method to use.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        raise NotImplementedError


class BinaryCrossEntropyWithLogitsLoss(BinaryClassificationLoss):
    """
    Callable class that computes binary cross entropy.
    """

    def __init__(
            self,
            weight: float = 0.5,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        weight : float
            The weight attributed to class 1 (in [0, 1]).
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name="BinaryCrossEntropyLoss", reduction=reduction, weight=weight)

        if self.reduction not in (LossReduction.MEAN.value, LossReduction.SUM.value):
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
            pos_weight=tensor([1 - self.scaling_factor, self.scaling_factor]),
            reduction="none"
        )

        return loss(pred, targets)


class DICELoss(SegmentationLoss):
    """
    Callable class that computes the DICE loss.
    """

    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name="DICELoss", reduction=reduction)

        if self.reduction not in (LossReduction.MEAN.value, LossReduction.SUM.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['mean', 'sum'].")

    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor,
            reduction: Optional[Union[LossReduction, str]]
    ) -> Tensor:
        """
        Returns average Dice loss between two tensors.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted probabilities of being in class 1
        targets : Tensor
            (N,) tensor with ground truth
        reduction : Optional[Union[LossReduction, str]]
            Reduction method to use.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        loss = DiceLoss(sigmoid=True, reduction="none")

        return self.perform_reduction(loss(pred, targets), reduction)
