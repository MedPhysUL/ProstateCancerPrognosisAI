"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterObject` object.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Dict, Union

from optuna.trial import Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterObject(HyperparameterContainer):
    """
    An object hyperparameter.
    """

    def __init__(
            self,
            constructor: Callable,
            parameters: Dict[str, Union[Hyperparameter, HyperparameterContainer]]
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Callable
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            an object given the hyperparameters.
        parameters : Dict[str, Union[Hyperparameter, HyperparameterContainer]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        super().__init__(sequence=list(parameters.values()))
        self._constructor = constructor
        self._parameters = parameters

    def get_suggestion(
            self,
            trial: Trial
    ) -> object:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : object
            Optuna's current suggestion for this object hyperparameter.
        """
        self_copy = deepcopy(self)
        params = {name: hp.get_suggestion(trial) for name, hp in self_copy._parameters.items()}
        return self_copy._constructor(**params)

    def get_fixed_value(
            self,
            parameters: Dict[str, Any]
    ) -> object:
        """
        Gets the value of the hyperparameter container using the given parameters dictionary.

        Parameters
        ----------
        parameters : Dict[str, Any]
            A dictionary containing hyperparameters' values.

        Returns
        -------
        fixed_value : Any
            The fixed value of the hyperparameter container.
        """
        self.verify_params_keys(parameters)

        self_copy = deepcopy(self)
        constructor_params = {name: hp.get_fixed_value(parameters) for name, hp in self_copy._parameters.items()}
        return self_copy._constructor(**constructor_params)
