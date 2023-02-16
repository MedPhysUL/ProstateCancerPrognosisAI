"""
    @file:              categorical.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `CategoricalHyperparameter` object.
"""

from typing import Any, Iterable

from .hyperparameter import Hyperparameter


class CategoricalHyperparameter(Hyperparameter):
    """
    A categorical hyperparameter.
    """

    def __init__(
            self,
            name: str,
            values: Iterable[Any]
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        values : Iterable[Any]
            Search space of the hyperparameter, i.e. a list of the categorical values to try.
        """
        super().__init__(name=name)
        self.values = values
