"""
    @file:              loss.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `SingleTaskLoss` class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from torch import nanmean, nansum, Tensor

from .reduction import LossReduction


class SingleTaskLoss(ABC):
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
        self.reduction = LossReduction(reduction)
        self.name = name if name else f"{self.__class__.__name__}('reduction'='{self.reduction}')"

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
            reduction = LossReduction(reduction)

        if reduction == LossReduction.NONE:
            return x
        elif reduction == LossReduction.MEAN:
            return nanmean(x)
        elif reduction == LossReduction.SUM:
            return nansum(x)
