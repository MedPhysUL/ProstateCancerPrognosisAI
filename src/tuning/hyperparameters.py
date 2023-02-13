"""
    @file:              hyperparameters.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define classes related to hyperparameters.
"""

from enum import auto, StrEnum


class Distribution(StrEnum):
    """
    Stores possible types of distribution.
    """
    INT = auto()
    UNIFORM = auto()
    CATEGORICAL = auto()


class Range(StrEnum):
    """
    Stores possible hyperparameters' range types.
    """
    MIN = auto()
    MAX = auto()
    STEP = auto()
    VALUES = auto()
    VALUE = auto()


class Hyperparameter:
    """
    A class that represents a hyperparameter.
    """

    def __init__(
            self,
            name: str,
            distribution: str
    ) -> None:
        """
        Sets the name of the hp and the distribution from which the suggestion must be sampled.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        distribution : str
            Optuna distribution from which it must be sampled.
        """
        self.name = name
        self.distribution = distribution

    def __repr__(
            self
    ) -> str:
        return self.name


class CategoricalHyperparameter(Hyperparameter):
    """
    A class that defines a Categorical hyperparameter.
    """

    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        super().__init__(name=name, distribution=Distribution.CATEGORICAL)


class NumericalIntHyperparameter(Hyperparameter):
    """
    A class that defines a Numerical integer hyperparameter.
    """
    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        super().__init__(name=name, distribution=Distribution.INT)


class NumericalContinuousHyperparameter(Hyperparameter):
    """
    Numerical continuous hyperparameter
    """

    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        super().__init__(name=name, distribution=Distribution.UNIFORM)
