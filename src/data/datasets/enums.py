"""
    @file:              enums.py
    @Author:            Maxence Larose, Nicolas Raymond, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 06/2023

    @Description:       This file contains the 'Mask' and 'Split' enum class.
"""

from enum import auto, StrEnum


class Mask(StrEnum):
    """
    Stores the constant related to mask categories.
    """
    TRAIN = auto()
    VALID = auto()
    TEST = auto()
    INNER = auto()


class Split(StrEnum):
    """
    Stores the constant related to split categories.
    """
    INNER = auto()
    OUTER = auto()
