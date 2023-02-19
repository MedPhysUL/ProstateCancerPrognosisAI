"""
    @file:              torch.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `TorchObjective` class used for hyperparameters tuning.
"""

from os import cpu_count
from typing import Any, Callable, Dict, List

import ray

from .base import Objective
from ...data.datasets import ProstateCancerDataset
from ..hyperparameters import HyperparameterDict, HyperparameterObject


class TorchObjective(Objective):
    """
    Callable objective function to use with the tuner for TorchModel.
    """

    TRAINER_INSTANCE_KEY = "trainer"
    TRAIN_METHOD_PARAMS_KEY = "train"

    def __init__(
            self,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
            trainer_constructor_hps: HyperparameterObject,
            train_method_hps: HyperparameterDict,
            num_cpus: int = cpu_count(),
            num_gpus: int = 0
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Custom dataset containing all the data needed for our evaluations.
        masks : Dict[int, Dict[str, List[int]]]
            Dict with list of idx to use as train, valid and test masks.
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
        super().__init__(
            dataset=dataset,
            masks=masks,
            num_cpus=num_cpus,
            num_gpus=num_gpus
        )

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

    def _build_trial_runner(self) -> Callable:
        """
        Builds the function run in parallel for each set of hyperparameters and return the score.

        Returns
        -------
        run_function : Callable
            Function that train a single model using given masks and trial.
        """

        @ray.remote(num_cpus=self._num_cpus, num_gpus=self._num_gpus)
        def run_trial(
                masks: Dict[str, List[int]],
                hyperparameters: Dict[str, Any]
        ) -> List[float]:
            """
            Trains a single model using given masks and hyperparameters.

            Parameters
            ----------
            masks : Dict[str, List[int]]
                Dictionary with list of integers for train, valid and test mask.
            hyperparameters : Dict[str, Any]
                Suggested hyperparameters for this trial.

            Returns
            -------
            scores : List[float]
                List of score values.
            """
            # We retrieve the trainer instance and the train method parameters from the suggested hyperparameters
            trainer_instance = hyperparameters[self.TRAINER_INSTANCE_KEY]
            train_method_params = hyperparameters[self.TRAIN_METHOD_PARAMS_KEY]

            # We create a copy of the current dataset and update its masks
            subset = self._get_subset(masks=masks)
            train_method_params[self.DATASET_KEY] = subset

            # We train the model using the suggested hyperparameters
            trainer_instance.train(**train_method_params)

            # We find the optimal threshold for each classification tasks
            trainer_instance.model.fix_thresholds_to_optimal_values(subset)

            # We calculate the scores on the different tasks
            test_set_scores = trainer_instance.model.scores_dataset(dataset=subset, mask=subset.test_mask)

            # We retrieve the score associated to the tuning metric
            scores = [test_set_scores[task.name][task.hps_tuning_metric.name] for task in subset.tasks]

            return scores

        return run_trial
