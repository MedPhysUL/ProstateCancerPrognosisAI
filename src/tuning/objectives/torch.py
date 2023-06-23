"""
    @file:              torch.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `TorchObjective` class used for hyperparameters tuning.
"""

from os import path
from typing import Any, Dict, List

from optuna.trial import FrozenTrial, Trial

from .base import ModelEvaluationContainer, Objective
from ..callbacks.containers import TuningCallbackList
from ...data.datasets import ProstateCancerDataset
from ..hyperparameters.containers import HyperparameterDict
from ..hyperparameters.torch import TrainerHyperparameter, TrainMethodHyperparameter
from ...training import Trainer


class TorchObjective(Objective):
    """
    Callable objective function to use with the tuner for TorchModel.
    """

    TRAINER_INSTANCE_KEY = "trainer"
    TRAIN_METHOD_PARAMS_KEY = "train"

    def __init__(
            self,
            trainer_hyperparameter: TrainerHyperparameter,
            train_method_hyperparameter: TrainMethodHyperparameter
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        trainer_hyperparameter : TrainerHyperparameter
            Trainer constructor hyperparameters.
        train_method_hyperparameter : TrainMethodHyperparameter
            Train method hyperparameters.
        """
        super().__init__()

        self._hyperparameters = HyperparameterDict(
            {
                self.TRAINER_INSTANCE_KEY: trainer_hyperparameter,
                self.TRAIN_METHOD_PARAMS_KEY: train_method_hyperparameter
            }
        )

    @property
    def hyperparameters(self) -> HyperparameterDict:
        """
        Hyperparameters.

        Returns
        -------
        hyperparameters : HyperparameterDict
            Dictionary containing hyperparameters.
        """
        return self._hyperparameters

    def __call__(
            self,
            trial: Trial,
            callbacks: TuningCallbackList,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
    ) -> List[float]:
        """
        Extracts hyperparameters suggested by optuna and executes the parallel inner loops.

        Parameters
        ----------
        trial : Trial
            Optuna trial.
        callbacks : TuningCallbackList
            Callbacks to use during tuning.
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        masks : Dict[int, Dict[str, List[int]]]
            Dictionary of inner loops masks, i.e a dictionary with list of idx to use as train, valid and test masks.

        Returns
        -------
        scores : List[float]
            List of task scores associated with the set of hyperparameters.
        """
        self._set_dataset(dataset)
        return super().__call__(trial, callbacks, dataset, masks)

    def exec_best_model_evaluation(
            self,
            best_trial: FrozenTrial,
            dataset: ProstateCancerDataset,
            path_to_save: str
    ) -> ModelEvaluationContainer:
        """
        Evaluates the best model.

        Parameters
        ----------
        best_trial : FrozenTrial
            Optuna trial.
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        path_to_save : str
            Path to save.

        Returns
        -------
        model_evaluation : ModelEvaluationContainer
            Model evaluation.
        """
        self._set_dataset(dataset)
        return super().exec_best_model_evaluation(best_trial, dataset, path_to_save)

    def _set_dataset(self, dataset: ProstateCancerDataset):
        """
        Sets dataset instance.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset.
        """
        self.hyperparameters[self.TRAIN_METHOD_PARAMS_KEY].dataset = dataset

    def _test_hyperparameters(
            self,
            dataset: ProstateCancerDataset,
            hyperparameters: Dict[str, Any],
            path_to_save: str
    ) -> ModelEvaluationContainer:
        """
        Tests hyperparameters and returns the train, valid and test scores.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        hyperparameters : Dict[str, Any]
            Suggested hyperparameters for this trial.
        path_to_save : str
            Path to the directory to save the current scores.

        Returns
        -------
        model_evaluation : ModelEvaluationContainer
            Model evaluation.
        """
        # We retrieve the trainer instance and the train method parameters from the suggested hyperparameters
        trainer_instance = hyperparameters[self.TRAINER_INSTANCE_KEY]
        train_method_params = hyperparameters[self.TRAIN_METHOD_PARAMS_KEY]

        # We prepare the trainer instance
        self._set_checkpoint_path(path_to_save, trainer_instance)
        train_method_params[self.DATASET_KEY] = dataset

        # We train the model using the suggested hyperparameters
        model_instance, _ = trainer_instance.train(**train_method_params)

        return self._get_model_evaluation(model_instance, dataset)

    @staticmethod
    def _set_checkpoint_path(path_to_directory: str, trainer: Trainer):
        """
        Sets checkpoint path.

        Parameters
        ----------
        path_to_directory : str
            Path to directory.
        trainer : Trainer
            Trainer.
        """
        if trainer.checkpoint:
            trainer.checkpoint.path_to_checkpoint_folder = path.join(
                path_to_directory,
                path.basename(trainer.checkpoint.path_to_checkpoint_folder)
            )
