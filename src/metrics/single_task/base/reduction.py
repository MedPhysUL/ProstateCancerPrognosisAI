"""
    @file:              reduction.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `MetricReduction` class.
"""

from enum import auto
from strenum import LowercaseStrEnum


class MetricReduction(LowercaseStrEnum):
    """
    Custom enum for reduction methods used to reduce the scores of multiple samples to a common score.
    """
    NONE = auto()
    MEAN = auto()
    SUM = auto()
    GEOMETRIC_MEAN = auto()
