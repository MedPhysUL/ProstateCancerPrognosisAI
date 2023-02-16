"""
    @file:              integer.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `IntegerHyperparameter` object.
"""

from optuna.trial import Trial

from .base import Hyperparameter


class IntegerHyperparameter(Hyperparameter):
    """
    A numerical integer hyperparameter.
    """

    def __init__(
            self,
            name: str,
            low: int,
            high: int,
            step: int = 1,
            log: bool = False
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        low : int
            Lower endpoint of the range of suggested values.
        high : int
            Upper endpoint of the range of suggested values.
        step : int
            A step of discretization. The step != 1 and log arguments cannot be used at the same time. To set the step
            argument 'step'>=2, set the log argument to False.
        log : bool
            A flag to sample the value from the log domain or not.
        """
        super().__init__(name=name)
        self.low = low
        self.high = high
        self.step = step
        self.log = log

    def get_suggestion(
            self,
            trial: Trial
    ) -> int:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : int
            Optuna's current suggestion for this hyperparameter.
        """
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)
