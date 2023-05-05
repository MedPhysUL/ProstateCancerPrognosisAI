"""
    @file:              mask.py
    @Author:            Maxence Larose, Nicolas Raymond, Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file contains the 'Mask' enum class.
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
