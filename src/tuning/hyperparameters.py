"""
    @file:              hyperparameters.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define classes related to hyperparameters.
"""


class Distribution:
    """
    Stores possible types of distribution.
    """
    INT: str = "int"                # Int uniform
    UNIFORM: str = "uniform"
    CATEGORICAL: str = "categorical"


class Range:
    """
    Stores possible hyperparameters' range types.
    """
    MIN: str = "min"
    MAX: str = "max"
    STEP: str = "step"
    VALUES: str = "values"
    VALUE: str = "value"


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


class CategoricalHP(Hyperparameter):
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


class NumericalIntHP(Hyperparameter):
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


class NumericalContinuousHP(Hyperparameter):
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
