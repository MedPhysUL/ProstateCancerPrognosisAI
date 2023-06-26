"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterObject` object.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

from .dict import HyperparameterDict


class HyperparameterObject(HyperparameterDict):
    """
    An object hyperparameter.
    """

    def __init__(
            self,
            constructor: Callable,
            parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Callable
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            an object given the hyperparameters.
        parameters : Optional[Dict[str, Any]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        if parameters:
            parameters = parameters
        else:
            parameters = {}

        super().__init__(container=parameters)
        self.constructor = constructor

    def build(
            self,
            suggestion: Dict[str, Any]
    ) -> object:
        """
        Builds hyperparameter given a suggestion and returns the hyperparameter instance.

        Parameters
        ----------
        suggestion : Dict[str, Any]
            Hyperparameters suggestion.

        Returns
        -------
        hyperparameter_instance : object
            Hyperparameter instance.
        """
        constructor_params = super().build(suggestion)
        return self.constructor(**constructor_params)
