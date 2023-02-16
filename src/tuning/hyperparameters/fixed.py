"""
    @file:              fixed.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `FixedHyperparameter` object.
"""

from typing import Any

from optuna.trial import Trial

from .base import Hyperparameter


class FixedHyperparameter(Hyperparameter):
    """
    A fixed hyperparameter.
    """

    def __init__(
            self,
            value: Any
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        value : Any
            The hyperparameter fixed value.
        """
        super().__init__(name=self.__class__.__name__)
        self.value = value

    def get_suggestion(
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
