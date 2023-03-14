"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterObject` object.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional

from optuna.trial import FrozenTrial, Trial

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
        constructor_params = self._get_params(lambda hp: hp.suggest(trial))
        return self.constructor(**constructor_params)

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
        constructor_params = self._get_params(lambda hp: hp.retrieve_suggestion(trial))
        return self.constructor(**constructor_params)
