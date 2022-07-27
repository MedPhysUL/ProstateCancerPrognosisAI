"""
    @file:              base_model.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains the skeleton for the various implemented learning models.

"""

from abc import ABC


class BaseModel(ABC):
    """
    A structure for the various learning models implemented.
    """

    def fit(self):
        """
        Trains the parameters of the models.
        """
        pass

    def predict(self):
        """
        Predicts the label of a given features set.
        """
        pass








