"""
    @file:              metric.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Direction`, `Metric` and `MetricReduction` classes.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

from torch import any, nanmean, pow, prod, nansum, Tensor


class Direction(Enum):
    """
    Custom enum for optimization directions
    """
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

    def __iter__(self):
        return iter([self.MAXIMIZE, self.MINIMIZE])


class MetricReduction(Enum):
    """
    Custom enum for reduction methods used to reduce the scores of multiple samples to a common score.
    """
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    GEOMETRIC_MEAN = "geometric_mean"


class Metric(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as optimization or evaluation metrics.
    """

    def __init__(
            self,
            direction: Union[Direction, str],
            name: str,
            reduction: Union[MetricReduction, str],
            n_digits: int = 5
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        direction : Union[Direction, str]
            Whether the metric needs to be "maximize" or "minimize".
        name : str
            Name of the metric.
        reduction : Union[MetricReduction, str]
            Reduction method to use.
        n_digits : int
            Number of digits kept.
        """
        self.direction = Direction(direction).value
        self.reduction = MetricReduction(reduction).value
        self.n_digits = n_digits

        if name:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}('reduction'={repr(self.reduction)}, 'n_digits'={n_digits})"

    @abstractmethod
    def __call__(
            self,
            *args,
            **kwargs
    ) -> float:
        """
        Gets metric value.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        raise NotImplementedError

    def perform_reduction(
            self,
            x: Union[float, Tensor],
            reduction: Optional[Union[MetricReduction, str]] = None
    ) -> float:
        """
        Gets metric value.

        Parameters
        ----------
        x : Union[float, Tensor]
            Float or (N, 1) tensor.
        reduction : Optional[Union[MetricReduction, str]]
            Reduction method to use. If None, we use the default reduction, i.e. self.reduction.

        Returns
        -------
        metric : float
            Rounded metric score.
        """
        if reduction is None:
            reduction = self.reduction
        else:
            reduction = MetricReduction(reduction).value

        if reduction == MetricReduction.NONE.value:
            if isinstance(x, float):
                return x
            else:
                return x.item()
        elif reduction == MetricReduction.MEAN.value:
            return nanmean(x).item()
        elif reduction == MetricReduction.SUM.value:
            return nansum(x).item()
        elif reduction == MetricReduction.GEOMETRIC_MEAN.value:
            filtered_x = x[~any(x.isnan(), dim=1)]
            return pow(prod(filtered_x), exponent=(1 / filtered_x.shape[0])).item()
