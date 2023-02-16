"""
    @file:              distribution.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to store possible types of hyperparameters distribution.
"""

from enum import auto, StrEnum


class Distribution(StrEnum):
    """
    Stores possible types of distribution.
    """
    INT = auto()
    UNIFORM = auto()
    CATEGORICAL = auto()
