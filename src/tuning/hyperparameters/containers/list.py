"""
    @file:              list.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterList` object.
"""

from typing import Any, List, Union

from optuna.trial import Trial

from .base import Hyperparameter, HyperparameterContainer


class HyperparameterList(HyperparameterContainer):
    """
    A hyperparameter list.
    """

    def __init__(
            self,
            container: List[Union[Hyperparameter, HyperparameterContainer]]
    ) -> None:
        """
        Hyperparameters list constructor.

        Parameters
        ----------
        container : List[Union[Hyperparameter, HyperparameterContainer]]
            List of hyperparameters.
        """
        super().__init__(sequence=container)

    def get_suggestion(
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
        return [hp.get_suggestion(trial) for hp in self._sequence]
