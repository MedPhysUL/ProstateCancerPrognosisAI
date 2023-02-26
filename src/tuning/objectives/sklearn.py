"""
    @file:              sklearn.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `SklearnObjective` class used for hyperparameters tuning.
"""

from os import cpu_count
from typing import Any, Callable, Dict

import ray

from .base import Objective, ScoreContainer
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

    def _build_inner_loop_runner(self) -> Callable:
        """
        Builds the function run in parallel for each set of hyperparameters and return the score.

        Returns
        -------
        run_inner_loop : Callable
            Function that train a single model using given masks and trial.
        """

        @ray.remote(num_cpus=self.num_cpus, num_gpus=self.num_gpus)
        def run_inner_loop(
                hyperparameters: Dict[str, Any]
        ) -> ScoreContainer:
            """
            Trains a single model using given masks and hyperparameters.

            Parameters
            ----------
            hyperparameters : Dict[str, Any]
                Suggested hyperparameters for this trial.

            Returns
            -------
            score : ScoreContainer
                Score values.
            """
            self.inner_loop_state.callbacks.on_inner_loop_start(self)

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

            self.inner_loop_state.score = score
            self.inner_loop_state.callbacks.on_inner_loop_end(self)

            return score

        return run_inner_loop
