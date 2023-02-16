"""
    @file:              object.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `ObjectHyperparameter` object.
"""

from __future__ import annotations
from copy import deepcopy
from typing import Callable, Dict, List

from optuna.trial import Trial

from .base import Hyperparameter
from .fixed import FixedHyperparameter


class ObjectHyperparameter(Hyperparameter):
    """
    An object hyperparameter.
    """

    def __init__(
            self,
            constructor: Callable,
            hyperparameters: Dict[str, Hyperparameter]
    ) -> None:
        """
        Sets attribute using parent's constructor and validates that 'hyperparameters' exclusively contains instances of
        `Hyperparameter`.

        Parameters
        ----------
        constructor : Callable
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            an object given the hyperparameters.
        hyperparameters : Dict[str, Hyperparameter]
            A dictionary of hyperparameters to initialize the object with. The keys are the names of the parameters used
            to build the given class constructor using its __init__ method.
        """
        assert all(isinstance(hp, Hyperparameter) for hp in hyperparameters.values()), (
            "All objects in 'hyperparameters' must be instances of 'Hyperparameter'."
        )
        super().__init__(name=constructor.__name__)
        self.hyperparameters = hyperparameters
        self._constructor = constructor

    @property
    def hyperparameters(self) -> Dict[str, Hyperparameter]:
        """
        'hyperparameters' property.

        Returns
        -------
        hyperparameters : Dict[str, Hyperparameter]
            A dictionary of hyperparameters to initialize the object with. The keys are the names of the parameters used
            to build the given class constructor using its __init__ method.
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: Dict[str, Hyperparameter]):
        """
        'hyperparameters' setter.

        Parameters
        ----------
        hyperparameters : Dict[str, Hyperparameter]
            A dictionary of hyperparameters to initialize the object with. The keys are the names of the parameters used
            to build the given class constructor using its __init__ method.
        """
        self._check_hyperparameter_names_uniqueness(hyperparameters)
        self._hyperparameters = hyperparameters

    def _check_hyperparameter_names_uniqueness(
            self,
            hyperparameters: Dict[str, Hyperparameter]
    ) -> List[str]:
        """
        Raise an assertion error if there is any hyperparameter with the same name.

        Returns
        -------
        names : List[str]
            The unique hyperparameters' names.
        """
        seen, duplicates = [], []

        for hp in hyperparameters.values():
            if isinstance(hp, FixedHyperparameter):
                continue
            elif isinstance(hp, ObjectHyperparameter):
                hp_names = self._check_hyperparameter_names_uniqueness(hp.hyperparameters)
                for hp_name in hp_names:
                    if hp_name in seen:
                        duplicates.append(hp_name)
                    else:
                        seen.append(hp_name)
            else:
                if hp.name in seen:
                    duplicates.append(hp.name)
                else:
                    seen.append(hp.name)

        assert not duplicates, f"Duplicates hyperparameter names are not allowed. Found duplicates {duplicates}."

        return seen

    def get_suggestion(
            self,
            trial: Trial
    ) -> object:
        """
        Gets optuna's suggestion.

        Parameters
        ----------
        trial : Trial
            Optuna's hyperparameter optimization trial.

        Returns
        -------
        suggestion : object
            Optuna's current suggestion for this object hyperparameter.
        """
        self_copy = deepcopy(self)
        params = {name: hp.get_suggestion(trial) for name, hp in self_copy.hyperparameters.items()}
        return self_copy._constructor(**params)
