"""
    @file:              float.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `FloatHyperparameter` object.
"""

from typing import Any, Dict, Optional

from optuna.trial import Trial

from .base import Hyperparameter


class FloatHyperparameter(Hyperparameter):
    """
    A numerical continuous (float) hyperparameter.
    """

    def __init__(
            self,
            name: str,
            low: float,
            high: float,
            step: Optional[float] = None,
            log: bool = False
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        low : float
            Lower endpoint of the range of suggested values.
        high : float
            Upper endpoint of the range of suggested values.
        step : Optional[float]
            A step of discretization. The step and log arguments cannot be used at the same time. To set the step
            argument to a float number, set the log argument to False.
        log : bool
            A flag to sample the value from the log domain or not.
        """
        super().__init__(name=name)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def suggest(
            self,
            trial: Trial
    ) -> float:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : float
            Optuna's current suggestion for this hyperparameter.
        """
        return trial.suggest_float(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)

    def retrieve_suggestion(
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
