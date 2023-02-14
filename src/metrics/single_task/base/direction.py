"""
    @file:              direction.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Direction` class.
"""

from enum import auto, StrEnum


class Direction(StrEnum):
    """
    Custom enum for optimization directions
    """
    MAXIMIZE = auto()
    MINIMIZE = auto()
