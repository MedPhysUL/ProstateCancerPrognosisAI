"""
    @file:              training.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 03/2023

    @Description:       This file is used to define all the hyperparameters related to the trainer.
"""

from typing import Any, Dict, Optional, Union

from torch import device as torch_device

from ....data.datasets import ProstateCancerDataset
from .callbacks import CheckpointHyperparameter, LearningAlgorithmHyperparameter
from ..containers import HyperparameterDict, HyperparameterList, HyperparameterObject
from ....models.torch.base import TorchModel
from ..optuna import FixedHyperparameter
from ....training import Trainer


class TorchModelHyperparameter(HyperparameterObject):
    """Subclass"""


class TrainerHyperparameter(HyperparameterObject):

    def __init__(
            self,
            batch_size: Union[int, FixedHyperparameter] = 16,
            checkpoint: CheckpointHyperparameter = None,
            device: Optional[torch_device] = None,
            exec_metrics_on_train: bool = True,
            n_epochs: Union[int, FixedHyperparameter] = 100,
            verbose: bool = True,
            **kwargs
    ) -> None:
        """
        Sets attribute using parent's constructor.

        Parameters
        ----------
        batch_size : Optional[int, FixedHyperparameter]
            Size of the batches in the training loader. Default is 16.
        checkpoint : CheckpointHyperparameter
             Checkpoint used to manage and create the checkpoints of a model during the training process.
        device : Optional[torch_device]
            Device to use for the training process. Default is the device of the model.
        exec_metrics_on_train : bool
            Whether to compute metrics on the training set. This is useful when you want to save time by not computing
            the metrics on the training set. Default is True.
        n_epochs : Optional[int, FixedHyperparameter]
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
            n_epochs=n_epochs,
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
            suggestion: Dict[str, Any]
    ) -> TorchModel:
        """
        Gets model instance.

        Parameters
        ----------
        suggestion: Dict[str, Any]
            Hyperparameter suggestion.

        Returns
        -------
        model : TorchModel
            TorchModel.
        """
        model_hp = self.container[self.MODEL_KEY]
        model_constructor_params = suggestion[self.MODEL_KEY]
        model = model_hp.build(model_constructor_params)
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

    def build(
            self,
            suggestion: Dict[str, Any]
    ) -> object:
        """
        Gets the hyperparameters.

        Parameters
        ----------
        suggestion: Dict[str, Any]
            Hyperparameter suggestion.

        Returns
        -------
        hyperparameter_instance : object
            Hyperparameter instance.
        """
        model = self._get_model_instance(suggestion)
        self._build_container(model)

        params = self._get_params(lambda hp, name: hp.build(suggestion[name]))
        params[self.MODEL_KEY] = model

        return params
