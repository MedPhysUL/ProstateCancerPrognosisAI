"""
    @file:              hyperparameter.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Hyperparameter` object.
"""


class Hyperparameter:
    """
    An abstract hyperparameter.
    """

    def __init__(
            self,
            name: str
    ) -> None:
        """
        Sets the name of the hp and the distribution from which the suggestion must be sampled.

        Parameters
        ----------
        name : str
            Name of the hyperparameter.
        """
        self.name = name
