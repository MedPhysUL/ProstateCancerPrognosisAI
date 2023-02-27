"""
    @file:              torch.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `TorchObjective` class used for hyperparameters tuning.
"""

from os import cpu_count, path
from typing import Any, Dict

from .base import Objective, ScoreContainer
from ...data.datasets import ProstateCancerDataset
from ..hyperparameters import HyperparameterDict, HyperparameterObject
from ...training import Trainer


class TorchObjective(Objective):
    """
    Callable objective function to use with the tuner for TorchModel.
    """

    TRAINER_INSTANCE_KEY = "trainer"
    TRAIN_METHOD_PARAMS_KEY = "train"

    def __init__(
            self,
            trainer_constructor_hps: HyperparameterObject,
            train_method_hps: HyperparameterDict,
            num_cpus: int = cpu_count(),
            num_gpus: int = 0
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        trainer_constructor_hps : HyperparameterObject
            Trainer constructor hyperparameters.
        train_method_hps : HyperparameterDict
            Train method hyperparameters.
        num_cpus : int
            The quantity of CPU cores to reserve for the tuning task. This parameter does not affect the device used
            for training the model during each trial. For now, we have to use all cpus per trial, otherwise 'Ray' thinks
            the tasks have to be parallelized and everything falls apart because the GPU memory is not big enough to
            hold 2 trials (models) at a time.
        num_gpus : int
            The quantity of GPUs to reserve for the tuning task. This parameter does not affect the device used for
            training the model during each trial.
        """
        super().__init__(num_cpus=num_cpus, num_gpus=num_gpus)

        self._hyperparameters = HyperparameterDict(
            {
                self.TRAINER_INSTANCE_KEY: trainer_constructor_hps,
                self.TRAIN_METHOD_PARAMS_KEY: train_method_hps
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

    def _test_hyperparameters(
            self,
            dataset: ProstateCancerDataset,
            hyperparameters: Dict[str, Any],
            path_to_save: str
    ) -> ScoreContainer:
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
        score : ScoreContainer
            Score values.
        """
        # We retrieve the trainer instance and the train method parameters from the suggested hyperparameters
        trainer_instance = hyperparameters[self.TRAINER_INSTANCE_KEY]
        train_method_params = hyperparameters[self.TRAIN_METHOD_PARAMS_KEY]

        # We prepare the trainer instance
        self._set_checkpoint_path(path_to_save, trainer_instance)
        train_method_params[self.DATASET_KEY] = dataset

        # We train the model using the suggested hyperparameters
        trainer_instance.train(**train_method_params)

        # We find the optimal threshold for each classification tasks
        trainer_instance.model.fix_thresholds_to_optimal_values(dataset)

        # We calculate the scores on the different tasks on the different sets
        train_set_scores = trainer_instance.model.scores_dataset(dataset, dataset.train_mask)
        valid_set_scores = trainer_instance.model.scores_dataset(dataset, dataset.valid_mask)
        test_set_scores = trainer_instance.model.scores_dataset(dataset, dataset.test_mask)
        score = ScoreContainer(train=train_set_scores, valid=valid_set_scores, test=test_set_scores)

        return score

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
