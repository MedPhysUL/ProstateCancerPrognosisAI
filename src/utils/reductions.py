"""
    @file:              reductions.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define the reduction methods used to reduce the scores of multiple samples
                        to a common float score.
"""

from abc import ABC, abstractmethod
from typing import Union

from torch import mean, pow, prod, sum, Tensor


class Reduction(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as reduction methods.
    """

    @abstractmethod
    def __call__(self, x: Union[float, Tensor]) -> float:
        """
        Gets score value.

        Returns
        -------
        score : float
            Score.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the reduction method.

        Returns
        -------
        name : str
            Name.
        """
        raise NotImplementedError


class Identity(Reduction):

    def __call__(self, x: Union[float, Tensor]) -> float:
        if isinstance(x, float):
            return x
        else:
            return x.item()

    @property
    def name(self) -> str:
        return ""


class Mean(Reduction):

    def __call__(self, x: Tensor) -> float:
        return mean(x).item()

    @property
    def name(self) -> str:
        return "Mean"


class Sum(Reduction):

    def __call__(self, x: Tensor) -> float:
        return sum(x).item()

    @property
    def name(self) -> str:
        return "Sum"


class GeometricMean(Reduction):

    def __call__(self, x: Tensor) -> float:
        return pow(prod(x), exponent=(1 / x.shape[0])).item()

    @property
    def name(self) -> str:
        return "GeometricMean"
