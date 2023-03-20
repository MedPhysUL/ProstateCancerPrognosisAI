"""
    @file:              dict.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterDict` object.
"""

from typing import Any, Callable, Dict

from optuna.trial import FrozenTrial, Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterDict(HyperparameterContainer):
    """
    A hyperparameter dict.
    """

    def __init__(
            self,
            container: Dict[str, Any]
    ) -> None:
        """
        Hyperparameters dict constructor.

        Parameters
        ----------
        container : Dict[str, Any]
            Dict of hyperparameters.
        """
        self.container = container
        super().__init__(sequence=list(self.container.values()))

    def __getitem__(self, key: str) -> Any:
        """
        Gets value corresponding to given key in dictionary.

        Parameters
        ----------
        key : str
            Key.

        Returns
        -------
        value : Any
            Value.
        """
        return self.container[key]

    def _get_params(
            self,
            hyperparameter_value_getter: Callable
    ) -> Dict[str, Any]:
        """
        Get the hyperparameters.

        Parameters
        ----------
        hyperparameter_value_getter : Callable
            Hyperparameter value getter.

        Returns
        -------
        params : Dict[str, Any]
            Parameters.
        """
        params = {}
        for name, hp in self.container.items():
            if isinstance(hp, (Hyperparameter, HyperparameterContainer)):
                params[name] = hyperparameter_value_getter(hp, name)
            else:
                params[name] = hp

        return params

    def build(
            self,
            suggestion: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get the hyperparameters.

        Parameters
        ----------
        suggestion : Dict[str, Any]
            Suggestion.

        Returns
        -------
        hyperparameters : Dict[str, Any]
            Hyperparameters.
        """
        return self._get_params(lambda hp, name: hp.build(suggestion[name]))

    def suggest(
            self,
            trial: Trial
    ) -> Dict[str, Any]:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : Dict[Any]
            Optuna's current suggestions for all hyperparameters in the dict.
        """
        return self._get_params(lambda hp, name: hp.suggest(trial))

    def retrieve_past_suggestion(
            self,
            trial: FrozenTrial
    ) -> Dict[str, Any]:
        """
        Gets the value of the hyperparameter using the given parameters dictionary.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna's hyperparameter optimization frozen trial.

        Returns
        -------
        fixed_value : Dict[str, Any]
            The fixed value of the hyperparameter.
        """
        self.verify_params_keys(trial)
        return self._get_params(lambda hp, name: hp.retrieve_past_suggestion(trial))
