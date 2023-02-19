"""
    @file:              objective.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Objective` class used for hyperparameters tuning.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from os import cpu_count
from typing import Any, Callable, Dict, List

from optuna.trial import FrozenTrial, Trial
import numpy as np
import ray

from ...data.datasets import ProstateCancerDataset
from ...data.processing.sampling import Mask
from ..hyperparameters import HyperparameterDict


class Objective(ABC):
    """
    Abstract objective class to use within the tuner.
    """

    DATASET_KEY = "dataset"

    def __init__(
            self,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
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
        num_cpus : int
            The quantity of CPU cores to reserve for the tuning task. This parameter does not affect the device used
            for training the model during each trial. For now, we have to use all cpus per trial, otherwise 'Ray' thinks
            the tasks have to be parallelized and everything falls apart because the GPU memory is not big enough to
            hold 2 trials (models) at a time.
        num_gpus : int
            The quantity of GPUs to reserve for the tuning task. This parameter does not affect the device used for
            training the model during each trial.
        """
        self._dataset = dataset
        self._masks = masks
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._run_trial = self._build_trial_runner()

    @property
    def dataset(self) -> ProstateCancerDataset:
        """
        Dataset.

        Returns
        -------
        dataset : ProstateCancerDataset
            Dataset.
        """
        return self._dataset

    def __call__(
            self,
            trial: Trial
    ) -> List[float]:
        """
        Extracts hyperparameters suggested by optuna and executes the parallel evaluations of the hyperparameters set.

        Parameters
        ----------
        trial : Trial
            Optuna trial.

        Returns
        -------
        scores : List[float]
            List of task scores associated with the set of hyperparameters.
        """
        # We get the hyperparameters suggestion
        suggested_hps = self.hyperparameters.get_suggestion(trial)

        # We execute parallel evaluations
        futures = [self._run_trial.remote(masks=m, hyperparameters=suggested_hps) for m in self._masks.values()]
        scores = ray.get(futures)

        # We take the mean of the scores
        return list(np.mean(scores, axis=0))

    @property
    @abstractmethod
    def hyperparameters(self) -> HyperparameterDict:
        """
        Hyperparameters.

        Returns
        -------
        hyperparameters : HyperparameterDict
            Dictionary containing hyperparameters.
        """
        raise NotImplementedError

    @abstractmethod
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
            raise NotImplementedError

        return run_trial

    def _get_subset(self, masks: Dict[str, List[int]]) -> ProstateCancerDataset:
        """
        Gets a subset of the dataset given masks.

        Parameters
        ----------
        masks : Dict[str, List[int]]
            Dictionary with list of integers for train, valid and test mask.

        Returns
        -------
        subset : ProstateCancerDataset
            Subset of the dataset.
        """
        subset = deepcopy(self._dataset)
        subset.update_masks(train_mask=masks[Mask.TRAIN], valid_mask=masks[Mask.VALID], test_mask=masks[Mask.TEST])

        return subset

    def get_tested_hyperparameters(
            self,
            trial: FrozenTrial
    ) -> Dict[str, Any]:
        """
        Gets tested hyperparameters in a dictionary given a frozen optuna trial.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna frozen trial.

        Returns
        -------
        dictionary : Dict[str, Any]
            Dictionary with hyperparameters' values.
        """
        return self.hyperparameters.get_tested_values(trial=trial)
