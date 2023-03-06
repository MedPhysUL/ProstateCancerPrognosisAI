"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterObject` object.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, Optional, Union

from optuna.trial import FrozenTrial, Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterObject(HyperparameterContainer):
    """
    An object hyperparameter.
    """

    def __init__(
            self,
            constructor: Callable,
            parameters: Optional[Dict[str, Union[Hyperparameter, HyperparameterContainer]]] = None
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Callable
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            an object given the hyperparameters.
        parameters : Optional[Dict[str, Union[Hyperparameter, HyperparameterContainer]]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        super().__init__(sequence=list(parameters.values()))
        self._constructor = constructor

        if parameters:
            self._parameters = parameters
        else:
            self._parameters = {}

    def suggest(
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
        params = {name: hp.suggest(trial) for name, hp in self_copy._parameters.items()}
        return self_copy._constructor(**params)

    def retrieve_suggestion(
            self,
            trial: FrozenTrial
    ) -> object:
        """
        Gets the value of the hyperparameter using the given parameters dictionary.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna's hyperparameter optimization frozen trial.

        Returns
        -------
        fixed_value : object
            The fixed value of the hyperparameter.
        """
        self.verify_params_keys(trial)

        self_copy = deepcopy(self)
        constructor_params = {name: hp.retrieve_suggestion(trial) for name, hp in self_copy._parameters.items()}
        return self_copy._constructor(**constructor_params)
