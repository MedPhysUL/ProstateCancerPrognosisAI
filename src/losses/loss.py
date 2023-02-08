"""
    @file:              loss.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Loss` and `LossReduction` classes.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from torch import nanmean, nansum, Tensor


class LossReduction(Enum):
    """
    Custom enum for reduction methods used to reduce the losses of multiple samples to a common loss.
    """
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


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
        self.reduction = LossReduction(reduction).value
        self.name = name if name else f"{self.__class__.__name__}('reduction'={repr(self.reduction)})"

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
            Reduction method to use. If None, we use the default reduction, i.e. self.reduction.

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
            return nanmean(x)
        elif reduction == LossReduction.SUM.value:
            return nansum(x)
