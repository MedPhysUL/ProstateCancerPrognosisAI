"""
    @file:              objective.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Objective` class used for hyperparameters tuning.
"""

from abc import ABC, abstractmethod
from os import cpu_count
from typing import Any, Callable, Dict, List

from optuna.trial import Trial
import ray

from ...callbacks.containers import TuningCallbackList
from .containers import ScoreContainer
from ....data.datasets import ProstateCancerDataset
from ....data.processing.sampling import Mask
from ...hyperparameters import HyperparameterDict
from .states import InnerLoopState, TrialState


class Objective(ABC):
    """
    Abstract objective class to use within the tuner.
    """

    DATASET_KEY = "dataset"

    def __init__(
            self,
            num_cpus: int = cpu_count(),
            num_gpus: int = 0
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        num_cpus : int
            The quantity of CPU cores to reserve for the tuning task. This parameter does not affect the device used
            for training the model during each trial. For now, we have to use all cpus per trial, otherwise 'Ray' thinks
            the tasks have to be parallelized and everything falls apart because the GPU memory is not big enough to
            hold 2 trials/models at a time.
        num_gpus : int
            The quantity of GPUs to reserve for the tuning task. This parameter does not affect the device used for
            training the model during each trial.
        """
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.inner_loop_state = InnerLoopState()
        self.trial_state = TrialState()

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
        callbacks.on_trial_start(self)

        suggested_hps = self.hyperparameters.get_suggestion(trial)

        futures = []
        for idx, mask in enumerate(masks.values()):
            dataset.update_masks(mask[Mask.TRAIN], mask[Mask.VALID], mask[Mask.TEST])
            self.inner_loop_state.callbacks = callbacks
            self.inner_loop_state.dataset = dataset
            self.inner_loop_state.idx = idx

            self._run_inner_loop = self._build_inner_loop_runner()
            score = self._run_inner_loop.remote(hyperparameters=suggested_hps)
            futures.append(score)

        self.trial_state.scores = ray.get(futures)

        tuning_metric_test_scores = [
            self.trial_state.statistics.test[task.name][task.hps_tuning_metric.name].mean
            for task in dataset.tasks
        ]

        callbacks.on_trial_end(self)

        return tuning_metric_test_scores

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
    def _build_inner_loop_runner(self) -> Callable:
        """
        Builds the function run in parallel for each inner split and return the score.

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
            scores : ScoreContainer
                Score values.
            """
            raise NotImplementedError

        return run_inner_loop
