"""
    @file:              integer.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `IntegerHyperparameter` object.
"""

from typing import Union

from .hyperparameter import Hyperparameter


class IntegerHyperparameter(Hyperparameter):
    """
    A numerical integer hyperparameter.
    """

    def __init__(
            self,
            name: str,
            minimum: Union[int, float],
            maximum: Union[int, float],
            step: Union[int, float]
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        minimum : Union[int, float]
            Minimum value of the search space of the hyperparameter.
        maximum : Union[int, float]
            Maximum value of the search space of the hyperparameter.
        step : Union[int, float]
            Size of the steps to perform in the search space.
        """
        super().__init__(name=name)
        self.minimum = minimum
        self.maximum = maximum
        self.step = step
