"""
    @file:              list.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterList` object.
"""

from typing import Any, Callable, List

from optuna.trial import FrozenTrial, Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterList(HyperparameterContainer):
    """
    A hyperparameter list.
    """

    def __init__(
            self,
            container: List[Any]
    ) -> None:
        """
        Hyperparameters list constructor.

        Parameters
        ----------
        container : List[Any]
            List of hyperparameters.
        """
        self.container = container
        super().__init__(sequence=self.container)

    def _get_params(
            self,
            hyperparameter_value_getter: Callable
    ) -> List[Any]:
        """
        Get the hyperparameters.

        Parameters
        ----------
        hyperparameter_value_getter : Callable
            Hyperparameter value getter.

        Returns
        -------
        params : List[Any]
            Parameters.
        """
        params = []
        for hp in self.sequence:
            if isinstance(hp, (Hyperparameter, HyperparameterContainer)):
                params.append(hyperparameter_value_getter(hp))
            else:
                params.append(hp)

        return params

    def suggest(
            self,
            trial: Trial
    ) -> List[Any]:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : List[Any]
            Optuna's current suggestions for all hyperparameters in the list.
        """
        return self._get_params(lambda hp: hp.suggest(trial))

    def retrieve_suggestion(
            self,
            trial: FrozenTrial
    ) -> List[Any]:
        """
        Gets the value of the hyperparameter using the given parameters dictionary.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna's hyperparameter optimization frozen trial.

        Returns
        -------
        fixed_value : List[Any]
            The fixed value of the hyperparameter.
        """
        self.verify_params_keys(trial)
        return self._get_params(lambda hp: hp.retrieve_suggestion(trial))
