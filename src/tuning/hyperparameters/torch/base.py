"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file is used to define the abstract `ModelDependantHyperparameter` and
                        `OptimizerDependantHyperparameter` object.
"""

from abc import ABC
from typing import Any, Callable, Dict, Optional, Type

from torch.optim import Optimizer

from ..containers import HyperparameterObject
from ....models.base.torch_model import TorchModel
from ..optuna.base import Hyperparameter


class ModelDependantHyperparameter(HyperparameterObject, ABC):

    PARAMS_KEY = "params"

    def __init__(
            self,
            constructor: Type[object],
            model_params_getter: Callable = lambda model: model.parameters(),
            parameters: Optional[Dict[str, Hyperparameter]] = None
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Type[object]
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            the regularizer given the hyperparameters.
        model_params_getter : Callable
            Model parameters getter.
        parameters : Optional[Dict[str, Hyperparameter]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        super().__init__(constructor=constructor, parameters=parameters)

        self._model = None
        self._model_params_getter = model_params_getter

    @property
    def model(self) -> TorchModel:
        """
        Model.

        Returns
        -------
        model : TorchModel
            Model.
        """
        return self._model

    @model.setter
    def model(self, model: TorchModel):
        """
        Sets model.

        Parameters
        ----------
        model : TorchModel
            Model
        """
        self._model = model

    def _get_params(
            self,
            hyperparameter_value_getter: Callable
    ) -> Dict[str, Any]:
        """
        Get the hyperparameters.

        Parameters
        ----------
        hyperparameter_value_getter : Callable
            Hyperparameter value getter.

        Returns
        -------
        params : Dict[str, Any]
            Parameters.
        """
        params = super()._get_params(hyperparameter_value_getter)
        params[self.PARAMS_KEY] = self._model_params_getter(self._model)
        return params


class OptimizerDependantHyperparameter(HyperparameterObject, ABC):

    OPTIMIZER_KEY = "optimizer"

    def __init__(
            self,
            constructor: Type[object],
            parameters: Optional[Dict[str, Hyperparameter]] = None
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Type[object]
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            the optimizer given the hyperparameters.
        parameters : Optional[Dict[str, Hyperparameter]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        super().__init__(constructor=constructor, parameters=parameters)
        self._optimizer = None

    @property
    def optimizer(self) -> Optimizer:
        """
        Optimizer.

        Returns
        -------
        optimizer : Optimizer
            Optimizer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer):
        """
        Sets optimizer.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer.
        """
        self._optimizer = optimizer

    def _get_params(
            self,
            hyperparameter_value_getter: Callable
    ) -> Dict[str, Any]:
        """
        Gets the hyperparameters.

        Parameters
        ----------
        hyperparameter_value_getter : Callable
            Hyperparameter value getter.

        Returns
        -------
        params : Dict[str, Any]
            Parameters.
        """
        params = super()._get_params(hyperparameter_value_getter)
        params[self.OPTIMIZER_KEY] = self._optimizer
        return params
