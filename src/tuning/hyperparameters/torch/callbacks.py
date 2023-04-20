"""
    @file:              callbacks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file is used to define all the hyperparameters related to training callbacks.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union

from torch.optim import Optimizer

from .base import ModelDependantHyperparameter, OptimizerDependantHyperparameter
from ..containers.object import HyperparameterObject
from ..optuna.fixed import FixedHyperparameter
from ....training.callbacks import Checkpoint, LearningAlgorithm


class CriterionHyperparameter(HyperparameterObject):
    """Subclass"""


class EarlyStopperHyperparameter(HyperparameterObject):
    """Subclass"""


class LRSchedulerHyperparameter(OptimizerDependantHyperparameter):
    """Subclass"""


class OptimizerHyperparameter(ModelDependantHyperparameter):
    """Subclass"""


class RegularizerHyperparameter(ModelDependantHyperparameter):

    def __init__(
            self,
            constructor: Type[object],
            model_params_getter: Callable = lambda model: model.named_parameters(),
            parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        constructor : Type[object]
            The class constructor (also named 'class blueprint' or 'class object'). This constructor is used to build
            the regularizer given the hyperparameters.
        model_params_getter : Callable
            Model parameters' getter.
        parameters : Optional[Dict[str, Any]]
            A dictionary of parameters to initialize the object with. The keys are the names of the parameters used to
            build the given class constructor using its __init__ method.
        """
        super().__init__(constructor, model_params_getter, parameters)


class LearningAlgorithmHyperparameter(ModelDependantHyperparameter):

    OPTIMIZER_KEY = "optimizer"
    LR_SCHEDULER_KEY = "lr_scheduler"
    REGULARIZER_KEY = "regularizer"

    def __init__(
            self,
            criterion: CriterionHyperparameter,
            optimizer: OptimizerHyperparameter,
            early_stopper: Optional[EarlyStopperHyperparameter] = None,
            lr_scheduler: Optional[LRSchedulerHyperparameter] = None,
            name: Optional[Union[str, FixedHyperparameter]] = None,
            regularizer: Optional[Union[RegularizerHyperparameter, List[RegularizerHyperparameter]]] = None,
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        criterion : CriterionHyperparameter
            Multi-task loss.
        optimizer : OptimizerHyperparameter
            A pytorch Optimizer.
        early_stopper : Optional[EarlyStopperHyperparameter]
            An early stopper.
        lr_scheduler : Optional[LRSchedulerHyperparameter]
            A pytorch learning rate scheduler.
        name : Optional[Union[str, FixedHyperparameter]]
            The name of the callback.
        regularizer : Optional[Union[RegularizerHyperparameter, List[RegularizerHyperparameter]]]
            Regularizer.
        """
        params = dict(
            criterion=criterion,
            optimizer=optimizer,
            early_stopper=early_stopper,
            lr_scheduler=lr_scheduler,
            name=name,
            regularizer=regularizer
        )
        super().__init__(constructor=LearningAlgorithm, parameters=params)

    def _get_optimizer_instance(
            self,
            suggestion: Dict[str, Any]
    ) -> Optimizer:
        """
        Gets the hyperparameters.

        Parameters
        ----------
        suggestion : Dict[str, Any]
            Hyperparameter suggestion.

        Returns
        -------
        optimizer : Optimizer
            Optimizer.
        """
        optimizer_hp = self.container[self.OPTIMIZER_KEY]
        optimizer_hp.model = self.model
        optimizer_constructor_params = suggestion[self.OPTIMIZER_KEY]
        return optimizer_hp.build(optimizer_constructor_params)

    def _build_container(
            self,
            optimizer: Optimizer
    ) -> None:
        """
        Builds the hyperparameter container, i.e. updates the optimizer and the model used by the hyperparameters in the
        container.

        Parameters
        ----------
        optimizer : Optimizer
            Current optimizer instance.
        """
        regularizer = self.container[self.REGULARIZER_KEY]
        if regularizer:
            regularizer.model = self.model

        lr_scheduler = self.container[self.LR_SCHEDULER_KEY]
        if lr_scheduler:
            lr_scheduler.optimizer = optimizer

    def build(
            self,
            suggestion: Dict[str, Any]
    ) -> object:
        """
        Builds hyperparameter given a suggestion and returns the hyperparameter instance.

        Parameters
        ----------
        suggestion : Dict[str, Any]
            Hyperparameters suggestion.

        Returns
        -------
        hyperparameter_instance : object
            Hyperparameter instance.
        """
        optimizer = self._get_optimizer_instance(suggestion)
        self._build_container(optimizer)

        constructor_params = self._get_params(lambda hp, name: hp.build(suggestion[name]))
        constructor_params[self.OPTIMIZER_KEY] = optimizer

        return self.constructor(**constructor_params)


class CheckpointHyperparameter(HyperparameterObject):

    def __init__(
            self,
            epoch_to_start_save: int = 0,
            name: Optional[str] = None,
            save_freq: int = 1,
            save_model_state: bool = False,
            verbose: bool = False
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        epoch_to_start_save : int
            The epoch at which to start saving checkpoints.
        name : Optional[str]
            The name of the callback.
        save_freq : int
            The frequency at which to save checkpoints. If 'save_freq' <= 0, only save at the end of the training.
        save_model_state : bool
            Whether to include model state in checkpoint.
        verbose : bool
            Whether to print out the trace of the checkpoint.
        """
        params = dict(
            epoch_to_start_save=epoch_to_start_save,
            name=name,
            save_freq=save_freq,
            save_model_state=save_model_state,
            verbose=verbose
        )
        super().__init__(constructor=Checkpoint, parameters=params)
