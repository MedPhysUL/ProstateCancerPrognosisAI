"""
    @file:              reduction.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `LossReduction` class.
"""

from enum import auto
from strenum import LowercaseStrEnum


class LossReduction(LowercaseStrEnum):
    """
    Custom enum for reduction methods used to reduce the losses of multiple samples to a common loss.
    """
    NONE = auto()
    MEAN = auto()
    SUM = auto()
