"""
    @file:              hyperparameter.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Hyperparameter` object.
"""

from abc import ABC, abstractmethod
from typing import Any

from optuna.trial import FrozenTrial, Trial


class Hyperparameter(ABC):
    """
    An abstract hyperparameter.
    """

    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets the name of the hp.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        self.name = name

    @abstractmethod
    def suggest(
            self,
            trial: Trial
    ) -> Any:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : Any
            Optuna's current suggestion for this hyperparameter.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_suggestion(
            self,
            trial: FrozenTrial
    ) -> Any:
        """
        Gets the value of the hyperparameter using the given parameters dictionary.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna's hyperparameter optimization frozen trial.

        Returns
        -------
        fixed_value : Any
            The fixed value of the hyperparameter.
        """
        raise NotImplementedError
