"""
    @file:              fixed.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `FixedHyperparameter` object.
"""

from typing import Any

from optuna.trial import FrozenTrial, Trial

from .base import Hyperparameter


class FixedHyperparameter(Hyperparameter):
    """
    A fixed hyperparameter.
    """

    def __init__(
            self,
            name: str,
            value: Any
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        value : Any
            The hyperparameter fixed value.
        """
        super().__init__(name=name)
        self.value = value

    def suggest(
            self,
            trial: Trial
    ) -> Any:
        """
        Gets optuna's suggestion. In this case, the suggestion is always the same and equals the given 'value'.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : Any
            Fixed value of this hyperparameter.
        """
        return self.value

    def retrieve_past_suggestion(
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
        return self.value
