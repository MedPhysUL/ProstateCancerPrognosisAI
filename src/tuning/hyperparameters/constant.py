"""
    @file:              constant.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `ConstantHyperparameter` object.
"""

from typing import Any

from .hyperparameter import Hyperparameter


class ConstantHyperparameter(Hyperparameter):
    """
    A constant hyperparameter.
    """

    def __init__(
            self,
            name: str,
            value: Any
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        value : Any
            The hyperparameter constant value.
        """
        super().__init__(name=name)
        self.value = value
