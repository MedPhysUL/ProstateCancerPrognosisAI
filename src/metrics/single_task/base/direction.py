"""
    @file:              direction.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Direction` class.
"""

from enum import auto
from strenum import LowercaseStrEnum


class Direction(LowercaseStrEnum):
    """
    Custom enum for optimization directions
    """
    NONE = auto()
    MINIMIZE = auto()
    MAXIMIZE = auto()
