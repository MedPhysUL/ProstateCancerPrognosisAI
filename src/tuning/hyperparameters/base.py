"""
    @file:              base.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Hyperparameter` object.
"""

from abc import ABC, abstractmethod
from typing import Any

from optuna.trial import Trial


class Hyperparameter(ABC):
    """
    An abstract hyperparameter.
    """

    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets the name of the hp and the distribution from which the suggestion must be sampled.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        self.name = name

    @abstractmethod
    def get_suggestion(
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
