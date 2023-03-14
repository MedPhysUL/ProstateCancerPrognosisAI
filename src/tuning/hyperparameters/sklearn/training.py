"""
    @file:              training.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define all the hyperparameters related to the sklearn model.
"""

from ..containers import HyperparameterDict, HyperparameterObject


class SklearnModelHyperparameter(HyperparameterObject):
    """Subclass"""


class FitMethodHyperparameter(HyperparameterDict):
    """Subclass"""
