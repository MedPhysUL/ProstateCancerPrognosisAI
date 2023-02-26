"""
    @file:              categorical.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `CategoricalHyperparameter` object.
"""

from typing import Any, Dict, Sequence

from optuna.trial import Trial

from .base import Hyperparameter


class CategoricalHyperparameter(Hyperparameter):
    """
    A categorical hyperparameter.
    """

    def __init__(
            self,
            name: str,
            choices: Sequence[Any]
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        choices : Sequence[Any]
            Parameter value candidates.
        """
        super().__init__(name=name)
        self.choices = choices

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
        return trial.suggest_categorical(name=self.name, choices=self.choices)

    def get_fixed_value(
            self,
            parameters: Dict[str, Any]
    ) -> Any:
        """
        Gets the value of the hyperparameter using the given parameters dictionary.

        Parameters
        ----------
        parameters : Dict[str, Any]
            A dictionary containing hyperparameters' values.

        Returns
        -------
        fixed_value : Any
            The fixed value of the hyperparameter.
        """
        return parameters[self.name]
