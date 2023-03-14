"""
    @file:              training.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file is used to define all the hyperparameters related to the trainer.
"""

from typing import Any, Callable, Dict, Optional, Union

from torch import device as torch_device

from .base import Hyperparameter
from ....training.callbacks import Checkpoint
from ....data.datasets import ProstateCancerDataset
from .callbacks import LearningAlgorithmHyperparameter
from ..containers.base import HyperparameterContainer
from ..containers import HyperparameterDict, HyperparameterList, HyperparameterObject
from ....models.base.torch_model import TorchModel
from ..optuna import FixedHyperparameter
from ....training import Trainer


class TorchModelHyperparameter(HyperparameterObject):
    """Subclass"""


class TrainerHyperparameter(HyperparameterObject):

    def __init__(
            self,
            batch_size: Union[int, FixedHyperparameter] = 8,
            checkpoint: Checkpoint = None,
            device: Optional[torch_device] = None,
            exec_metrics_on_train: bool = True,
            max_epochs: Union[int, FixedHyperparameter] = 100,
            verbose: bool = True,
            **kwargs
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        batch_size : Optional[int, FixedHyperparameter]
            Size of the batches in the training loader. Default is 8.
        checkpoint : Checkpoint
             Checkpoint used to manage and create the checkpoints of a model during the training process.
        device : Optional[torch_device]
            Device to use for the training process. Default is the device of the model.
        exec_metrics_on_train : bool
            Whether to compute metrics on the training set. This is useful when you want to save time by not computing
            the metrics on the training set. Default is True.
        max_epochs : Optional[int, FixedHyperparameter]
            Maximum number of epochs for training. Default is 100.
        verbose : bool
            Whether to print out the trace of the trainer.
        **kwargs : dict
            x_transform : Module
                Transform to apply to the input data before passing it to the model.
            y_transform : Module
                Transform to apply to the target data before passing it to the model.
        """
        params = dict(
            batch_size=batch_size,
            checkpoint=checkpoint,
            device=device,
            exec_metrics_on_train=exec_metrics_on_train,
            max_epochs=max_epochs,
            verbose=verbose,
            **kwargs
        )
        super().__init__(constructor=Trainer, parameters=params)


class TrainMethodHyperparameter(HyperparameterDict):

    LEARNING_ALGORITHMS_KEY = "learning_algorithms"
    MODEL_KEY = "model"

    def __init__(
            self,
            model: TorchModelHyperparameter,
            learning_algorithms: Union[LearningAlgorithmHyperparameter, HyperparameterList]
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        model : TorchModelHyperparameter
            Model to train.
        learning_algorithms : Union[LearningAlgorithmHyperparameter, HyperparameterList]
            The learning algorithm callbacks.
        """
        learning_algorithms = self._initialize_learning_algorithms(learning_algorithms)
        params = dict(model=model, learning_algorithms=learning_algorithms)
        super().__init__(container=params)

        self._dataset = None

    @staticmethod
    def _initialize_learning_algorithms(
            learning_algorithms: Union[LearningAlgorithmHyperparameter, HyperparameterList]
    ):
        """
        Initializes learning algorithms and returns the initialized version.

        Parameters
        ----------
        learning_algorithms : Union[LearningAlgorithmHyperparameter, HyperparameterList]
            The learning algorithm callbacks.
        """
        if isinstance(learning_algorithms, LearningAlgorithmHyperparameter):
            learning_algorithms = HyperparameterList([learning_algorithms])
        elif isinstance(learning_algorithms, HyperparameterList):
            assert all([isinstance(algo, LearningAlgorithmHyperparameter) for algo in learning_algorithms.sequence]), (
                "All learning algorithms must be instances of `LearningAlgorithmHyperparameter`."
            )
        else:
            raise AssertionError(
                f"Parameter 'learning_algorithms' must either of type `LearningAlgorithmHyperparameter` or "
                f"`HyperparameterList[LearningAlgorithmHyperparameter]`. Found type {type(learning_algorithms)}."
            )

        return learning_algorithms

    @property
    def dataset(self) -> ProstateCancerDataset:
        """
        Dataset.

        Returns
        -------
        dataset : ProstateCancerDataset
            ProstateCancerDataset.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: ProstateCancerDataset):
        """
        Sets dataset.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            ProstateCancerDataset.
        """
        self._dataset = dataset

    def _get_model_instance(
            self,
            hyperparameter_value_getter: Callable
    ) -> TorchModel:
        """
        Gets model instance.

        Parameters
        ----------
        hyperparameter_value_getter : Callable
            Hyperparameter value getter.

        Returns
        -------
        model : TorchModel
            TorchModel.
        """
        model = hyperparameter_value_getter(self.container[self.MODEL_KEY])
        model.build(self._dataset)
        return model

    def _build_container(
            self,
            model: TorchModel
    ) -> None:
        """
        Builds the hyperparameter container, i.e. updates the model used by the hyperparameters in the container.

        Parameters
        ----------
        model : TorchModel
            Current model instance.
        """
        for learning_algorithm in self.container[self.LEARNING_ALGORITHMS_KEY].sequence:
            learning_algorithm.model = model

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
        model = self._get_model_instance(hyperparameter_value_getter)
        self._build_container(model)

        params = dict(model=model)
        for key, hp in self.container.items():
            if key not in params.keys():
                if isinstance(hp, (Hyperparameter, HyperparameterContainer)):
                    params[key] = hyperparameter_value_getter(hp)
                else:
                    params[key] = hp

        return params
