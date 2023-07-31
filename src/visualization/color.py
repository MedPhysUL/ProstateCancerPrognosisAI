"""
    @file:              color.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file contains the Color enum class.
"""

from enum import StrEnum


class LightColor(StrEnum):
    BLUE = "#9DC9E2"
    SAND = "#FECFA1"
    GREEN = "#B2DAAC"
    RED = "#FFA99C"
    PINK = "#CC87B3"
    PURPLE = "#9C99C6"


class Color(StrEnum):
    BLUE = "#70A7C7"
    SAND = "#F1B88C"
    GREEN = "#87C07E"
    RED = "#D87363"
    PINK = "#AF598F"
    PURPLE = "#716DA8"


class DarkColor(StrEnum):
    BLUE = "#4F609C"
    SAND = "#B39171"
    GREEN = "#226258"
    RED = "#9D343C"
    PINK = "#A1437E"
    PURPLE = "#4C3264"
