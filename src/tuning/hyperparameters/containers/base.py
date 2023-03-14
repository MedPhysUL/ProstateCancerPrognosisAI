"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `HyperparameterContainer` object.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Sequence

from optuna.trial import FrozenTrial, Trial

from ..optuna import FixedHyperparameter
from ..optuna.base import Hyperparameter


class HyperparameterContainer(ABC):
    """
    An abstract hyperparameter container.
    """

    def __init__(
            self,
            sequence: Sequence[Any]
    ) -> None:
        """
        Initializes the hyperparameter container.

        Parameters
        ----------
        sequence : Sequence[Any]
            A sequence of hyperparameters.
        """
        self.sequence = sequence
        self._set_hyperparameters()

    @property
    def hyperparameters(self) -> List[Hyperparameter]:
        """
        List of hyperparameters contained in the sequence.

        Returns
        -------
        hps : List[Hyperparameter]
            Hyperparameters.
        """
        return self._hyperparameters

    @property
    def fixed_hyperparameters(self) -> List[FixedHyperparameter]:
        """
        List of fixed hyperparameters contained in the sequence.

        Returns
        -------
        fixed_hps : List[FixedHyperparameter]
            Fixed hyperparameters.
        """
        return [hp for hp in self.hyperparameters if isinstance(hp, FixedHyperparameter)]

    @property
    def tunable_hyperparameters(self) -> List[Hyperparameter]:
        """
        List of tunable hyperparameters contained in the sequence.

        Returns
        -------
        tunable_hps : List[Hyperparameter]
            Tunable hyperparameters.
        """
        return [
            hp for hp in self.hyperparameters
            if isinstance(hp, Hyperparameter) and not isinstance(hp, FixedHyperparameter)
        ]

    def _set_hyperparameters(self):
        """
        Sets list of hyperparameters and checks hyperparameters names uniqueness.
        """
        hyperparameters = []
        for hp in self.sequence:
            if isinstance(hp, Hyperparameter):
                hyperparameters += [hp]
            elif isinstance(hp, HyperparameterContainer):
                hyperparameters += hp.hyperparameters

        self._hyperparameters = hyperparameters
        self._check_hyperparameters_names_uniqueness()

    def _check_hyperparameters_names_uniqueness(self):
        """
        Raises an assertion error if there is any hyperparameter with the same name.
        """
        seen = set()
        duplicates = [hp.name for hp in self.hyperparameters if hp.name in seen or seen.add(hp.name)]

        assert not duplicates, f"Duplicate hyperparameters names are not allowed. Found duplicates {duplicates}."

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
            Optuna's hyperparameters optimization trial.

        Returns
        -------
        suggestion : Any
            Optuna's current suggestion for the hyperparameters.
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

    def verify_params_keys(
            self,
            trial: FrozenTrial
    ):
        """
        Verify that the params dictionary sets all tunable hyperparameters.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna's hyperparameter optimization frozen trial.
        """
        assert all(hp.name in trial.params.keys() for hp in self.tunable_hyperparameters), (
            f"'params' must set all hyperparameter values of the current container."
        )
