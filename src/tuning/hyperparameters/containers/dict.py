"""
    @file:              dict.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterDict` object.
"""

from typing import Any, Dict, Union

from optuna.trial import Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterDict(HyperparameterContainer):
    """
    A hyperparameter dict.
    """

    def __init__(
            self,
            container: Dict[str, Union[Hyperparameter, HyperparameterContainer]]
    ) -> None:
        """
        Hyperparameters dict constructor.

        Parameters
        ----------
        container : Dict[str, Union[Hyperparameter, HyperparameterContainer]]
            Dict of hyperparameters.
        """
        super().__init__(sequence=list(container.values()))
        self._container = container

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
        return {name: hp.suggest(trial) for name, hp in self._container.items()}

    def retrieve_suggestion(
            self,
            parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
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
        return {name: hp.retrieve_suggestion(parameters) for name, hp in self._container.items()}
