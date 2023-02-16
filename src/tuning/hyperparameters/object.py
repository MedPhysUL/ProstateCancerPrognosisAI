"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `ObjectHyperparameter` object.
"""

from typing import Callable, List

from .hyperparameter import Hyperparameter


class ObjectHyperparameter(Hyperparameter):
    """
    An object hyperparameter.
    """

    def __init__(
            self,
            constructor: Callable,
            hyperparameters: List[Hyperparameter],
            name: str
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Callable
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            an object given the hyperparameters.
        hyperparameters : List[Hyperparameter]
            A list of hyperparameters to initialize the object with.
        name : str
            Name of the current object hyperparameter.
        """
        assert all(isinstance(hp, Hyperparameter) for hp in hyperparameters), (
            "All objects in 'hyperparameters' must be instances of 'Hyperparameter'."
        )

        super().__init__(name=name)
        self.constructor = constructor
        self.hyperparameters = hyperparameters
