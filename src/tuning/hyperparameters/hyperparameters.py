"""
    @file:              hyperparameters.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define hyperparameters categories.
"""

from .spaces import Space
from .distribution import Distribution


class Hyperparameter:
    """
    A class that represents a hyperparameter.
    """

    def __init__(
            self,
            distribution: Distribution,
            name: str,
            space: Space
    ) -> None:
        """
        Sets the name of the hp and the distribution from which the suggestion must be sampled.

        Parameters
        ----------
        distribution : str
            Optuna distribution from which it must be sampled.
        name : str
            Name of the hyperparameter.
        space : Space
            Search space of the hyperparameter
        """
        self.space = space
        self.name = name
        self.distribution = distribution


class CategoricalHyperparameter(Hyperparameter):
    """
    A class that defines a Categorical hyperparameter.
    """

    def __init__(
            self,
            name: str,
            space: Space
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        space : Space
            Search space of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.CATEGORICAL, space=space)


class NumericalIntHyperparameter(Hyperparameter):
    """
    A class that defines a Numerical integer hyperparameter.
    """
    def __init__(
            self,
            name: str,
            space: Space
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        space : Space
            Search space of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.INT, space=space)


class NumericalContinuousHyperparameter(Hyperparameter):
    """
    Numerical continuous hyperparameter
    """

    def __init__(
            self,
            name: str,
            space: Space
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        space : Space
            Search space of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.UNIFORM, space=space)
