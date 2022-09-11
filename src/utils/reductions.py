"""
    @file:              reductions.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define the reduction methods used to reduce the scores of multiple samples
                        to a common float score.
"""

from enum import Enum


class MetricReduction(Enum):
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
    GEOMETRIC_MEAN = "geometric_mean"


class LossReduction(Enum):
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"
