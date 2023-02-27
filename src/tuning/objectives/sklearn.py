"""
    @file:              sklearn.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `SklearnObjective` class used for hyperparameters tuning.
"""

from os import cpu_count
from typing import Any, Dict

from .base import ModelEvaluationContainer, Objective, ScoreContainer
from ...data.datasets import ProstateCancerDataset
from ..hyperparameters import HyperparameterDict, HyperparameterObject


class SklearnObjective(Objective):
    """
    Callable objective function to use with the tuner for SklearnModel.
    """

    MODEL_INSTANCE_KEY = "model"
    FIT_METHOD_PARAMS_KEY = "fit"

    def __init__(
            self,
            model_constructor_hps: HyperparameterObject,
            fit_method_hps: HyperparameterDict,
            num_cpus: int = cpu_count(),
            num_gpus: int = 0
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        model_constructor_hps : HyperparameterObject
            Model constructor hyperparameters.
        fit_method_hps : HyperparameterDict
            Fit method hyperparameters.
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
                self.MODEL_INSTANCE_KEY: model_constructor_hps,
                self.FIT_METHOD_PARAMS_KEY: fit_method_hps
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
        # We retrieve the model instance and the fit method parameters from the suggested hyperparameters
        model_instance = hyperparameters[self.MODEL_INSTANCE_KEY]
        fit_method_params = hyperparameters[self.FIT_METHOD_PARAMS_KEY]
        dataset = self.inner_loop_state.dataset
        fit_method_params[self.DATASET_KEY] = dataset

        # We train the model using the suggested hyperparameters
        model_instance.fit(**fit_method_params)

        # We find the optimal threshold for each classification tasks
        model_instance.fix_thresholds_to_optimal_values(dataset)

        # We calculate the scores on the different tasks on the different sets
        train_set_scores = model_instance.scores_dataset(dataset, dataset.train_mask)
        valid_set_scores = model_instance.scores_dataset(dataset, dataset.valid_mask)
        test_set_scores = model_instance.scores_dataset(dataset, dataset.test_mask)
        score = ScoreContainer(train=train_set_scores, valid=valid_set_scores, test=test_set_scores)

        return ModelEvaluationContainer(trained_model=model_instance, score=score)
