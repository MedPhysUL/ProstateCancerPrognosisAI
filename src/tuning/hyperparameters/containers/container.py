"""
    @file:              container.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `HyperparameterContainer` object.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Union

from optuna.trial import Trial

from ..base import Hyperparameter
from ..fixed import FixedHyperparameter


class HyperparameterContainer(ABC):
    """
    An abstract hyperparameter container.
    """

    def __init__(
            self,
            sequence: Sequence[Union[Hyperparameter, HyperparameterContainer]]
    ) -> None:
        """
        Initialize the hyperparameter container.

        Parameters
        ----------
        sequence : Sequence[Union[Hyperparameter, HyperparameterContainer]]
            A sequence of hyperparameters.
        """
        assert all(isinstance(hp, (Hyperparameter, HyperparameterContainer)) for hp in sequence), (
            "All objects in 'sequence' must be instances of 'Hyperparameter' or 'HyperparameterContainer'."
        )
        self._sequence = sequence
        self._check_hyperparameter_names_uniqueness()

    def _check_hyperparameter_names_uniqueness(self):
        """
        Raises an assertion error if there is any hyperparameter with the same name.
        """
        seen = set()
        duplicates = [name for name in self.names if name in seen or seen.add(name)]

        assert not duplicates, f"Duplicates hyperparameter names are not allowed. Found duplicates {duplicates}."

    @property
    def names(self) -> List[str]:
        """
        The names the hyperparameters contained in the sequence.

        Returns
        -------
        names : List[str]
            Hyperparameter names.
        """
        names = []
        for hp in self._sequence:
            if isinstance(hp, FixedHyperparameter):
                pass
            elif isinstance(hp, HyperparameterContainer):
                names += hp.names
            else:
                names += [hp.name]

        return names

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
            Optuna's hyperparameters optimization trial.

        Returns
        -------
        suggestion : Any
            Optuna's current suggestion for the hyperparameters.
        """
        raise NotImplementedError
