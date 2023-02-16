"""
    @file:              spaces.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define search spaces of hyperparameters.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, Union


@dataclass(frozen=True)
class Space(ABC):
    """
    Specify the search space of a hyperparameter.
    """
    pass


@dataclass(frozen=True)
class MinMaxSpace(Space):
    """
    Specify the search space of a hyperparameter using minimum and maximum values (with a step if needed).
    """
    minimum: Union[int, float]
    maximum: Union[int, float]
    step: Union[int, float] = None


@dataclass(frozen=True)
class ValuesSpace(Space):
    """
    Specify the search space of a hyperparameter using a list of values.
    """
    values: Iterable[Any] = None
