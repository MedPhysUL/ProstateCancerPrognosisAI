"""
    @file:              dict.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `HyperparameterDict` object.
"""

from typing import Any, Dict, Union

from optuna.trial import Trial

from .container import Hyperparameter, HyperparameterContainer


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

    def get_suggestion(
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
        return {name: hp.get_suggestion(trial) for name, hp in self._container.items()}
