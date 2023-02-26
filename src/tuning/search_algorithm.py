"""
    @file:              search_algorithm.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `SearchAlgorithm` class used to search for the optimal set of
                        hyperparameters.
"""

from functools import partial
from typing import Dict, List

from optuna import create_study
from optuna.logging import FATAL, set_verbosity
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler
from optuna.study import Study
import ray

from .callbacks.containers import TuningCallbackList
from ..data.datasets import ProstateCancerDataset
from .objectives.base import Objective


class SearchAlgorithm:
    """
    Object in charge of searching for the optimal hyperparameters.
    """

    def __init__(
            self,
            sampler: BaseSampler,
            pruner: BasePruner = NopPruner,
            storage: str = "sqlite:///tuning_history.db"
    ):
        """
        Sets all protected and public attributes.

        Parameters
        ----------
        sampler : BaseSampler
            Optuna's sampler to use.
        pruner : BasePruner
            Optuna's pruner to use.
        storage : str
            Database URL. If this argument is set to None, in-memory storage is used.
        """
        self.pruner = pruner
        self.sampler = sampler
        self.storage = storage

    def _create_new_study(
            self,
            dataset: ProstateCancerDataset,
            name: str = None
    ) -> Study:
        """
        Creates a new optuna study.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Dataset.
        name: str
            Study name.

        Returns
        -------
        study : Study
            Study object.
        """
        directions = [task.hps_tuning_metric.direction for task in dataset.tasks]

        if len(directions) == 1:
            study = create_study(
                direction=directions[0],
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=name,
                storage=self.storage
            )
        else:
            study = create_study(
                directions=directions,
                sampler=self.sampler,
                pruner=self.pruner,
                study_name=name,
                storage=self.storage
            )

        return study

    def search(
            self,
            dataset: ProstateCancerDataset,
            callbacks: TuningCallbackList,
            masks: Dict[int, Dict[str, List[int]]],
            objective: Objective,
            n_trials: int = 100,
            study_name: str = None,
            verbose: bool = True
    ) -> Study:
        """
        Searches for the hyperparameters that optimize the objective function, using the TPE algorithm.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        callbacks : TuningCallbackList
            Callbacks to use during tuning.
        masks : Dict[int, Dict[str, List[int]]]
            Dictionary of inner loops masks, i.e a dictionary with list of idx to use as train, valid and test masks.
        objective : Objective
            The objective.
        n_trials : int
            Number of sets of hyperparameters tested.
        verbose : bool
            Whether we want optuna to show a progress bar.
        study_name : str
            Study's name. If this argument is set to None, a unique name is generated automatically.
        """
        # We check ray status
        ray_already_init = self._check_ray_status()

        study = self._create_new_study(dataset, study_name)

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        study.optimize(
            func=partial(func=objective.__call__, masks=masks, dataset=dataset, callbacks=callbacks),
            n_trials=n_trials,
            gc_after_trial=True,
            show_progress_bar=verbose
        )

        # We shutdown ray if it has been initialized in this function
        if not ray_already_init:
           ray.shutdown()

        return study

    @staticmethod
    def _check_ray_status() -> bool:
        """
        Checks if ray was already initialized and initialize it if it's not.

        Returns
        -------
        Whether ray was already initialized.
        """
        # We initialize ray if it is not initialized yet
        ray_was_init = True
        if not ray.is_initialized():
            ray_was_init = False
            ray.init()

        return ray_was_init
