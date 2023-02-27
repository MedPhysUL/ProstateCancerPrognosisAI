"""
    @file:              tuner.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     03/2022
    @Last modification: 02/2023

    @Description:      This file is used to define the `Tuner` class in charge of tuning the hyperparameters.
"""

from typing import Dict, List, Optional, Union

from numpy.random import seed as np_seed
import ray
from optuna.trial import FrozenTrial
from torch import manual_seed

from .callbacks.base import TuningCallback
from .callbacks.containers import TuningCallbackList
from ..data.datasets import Mask, ProstateCancerDataset
from .search_algorithm import Objective, SearchAlgorithm
from .states import BestModelState, OuterLoopState, StudyState, TuningState


class Tuner:
    """
    Object in charge of evaluating a model over multiple different data splits and tuning the hyperparameters.
    """

    SPLIT_PREFIX: str = "split"

    def __init__(
            self,
            search_algorithm: SearchAlgorithm,
            callbacks: Optional[Union[TuningCallback, TuningCallbackList, List[TuningCallback]]] = None,
            n_trials: int = 100,
            seed: Optional[int] = None,
            verbose: bool = False
    ):
        """
        Set protected and public attributes.

        Parameters
        ----------
        search_algorithm : SearchAlgorithm
            Search algorithm used to search for the optimal set of hyperparameters.
        callbacks : Optional[Union[TuningCallback, TuningCallbackList, List[TuningCallback]]]
            The tuning callbacks.
        n_trials : int
            Number of sets of hyperparameters tested.
        seed : Optional[int]
            Random state used for reproducibility.
        verbose : bool
            Whether to print out the trace of the tuner.
        """
        self.callbacks = callbacks
        self.n_trials = n_trials
        self.search_algorithm = search_algorithm
        self.seed = seed
        self.verbose = verbose

        self.best_model_state, self.outer_loop_state, self.study_state, self.tuning_state = None, None, None, None
        self._initialize_states()

    @property
    def callbacks(self) -> TuningCallbackList:
        """
        Callbacks to use during the training process.

        Returns
        -------
        callbacks : TuningCallbackList
            Callback list
        """
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: Optional[Union[TuningCallback, TuningCallbackList, List[TuningCallback]]]):
        """
        Sets the callbacks attribute as a CallbackList and sorts the callbacks in this list.

        Parameters
        ----------
        callbacks : Optional[Union[TuningCallback, TuningCallbackList, List[TuningCallback]]]
            The callbacks to set the callbacks attribute to.
        """
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, (TuningCallback, TuningCallbackList, list)):
            raise AssertionError(
                "'callbacks must be of type 'TuningCallback', 'TuningCallbackList' or 'List[TuningCallback]'."
            )
        if isinstance(callbacks, TuningCallback):
            callbacks = [callbacks]

        self._callbacks = TuningCallbackList(callbacks)
        self._callbacks.sort()

    def _initialize_states(self):
        """
        Initializes all states.
        """
        self.best_model_state = BestModelState()
        self.outer_loop_state = OuterLoopState()
        self.study_state = StudyState()
        self.tuning_state = TuningState()

    def tune(
            self,
            objective: Objective,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]]
    ) -> None:
        """
        Performs nested subsampling validations to evaluate a model and tune the hyperparameters.

        Parameters
        ----------
        objective : Objective
            The objective to optimize.
        dataset : ProstateCancerDataset
            Custom dataset containing the whole learning dataset needed for our evaluations.
        masks : Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]]
            Dict with list of idx to use as train, valid and test masks.
        """
        self._initialize_states()
        self._set_seed()
        ray.init()

        self.callbacks.on_tuning_start(self)
        self.outer_loop_state.scores = []
        for k, v in masks.items():
            self.outer_loop_state.idx = k

            train_mask, valid_mask, test_mask, inner_masks = v[Mask.TRAIN], v[Mask.VALID], v[Mask.TEST], v[Mask.INNER]
            dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            self.outer_loop_state.dataset = dataset

            self.callbacks.on_outer_loop_start(self)
            self._exec_study(dataset=dataset, inner_masks=inner_masks, objective=objective)

            dataset.update_masks(train_mask=train_mask, valid_mask=valid_mask, test_mask=test_mask)
            self._exec_best_model_evaluation(dataset=dataset, objective=objective)
            self.callbacks.on_outer_loop_end(self)

        # We shutdown ray
        ray.shutdown()
        self.callbacks.on_tuning_end(self)

    def _exec_study(
            self,
            dataset: ProstateCancerDataset,
            inner_masks: Dict[int, Dict[str, List[int]]],
            objective: Objective
    ):
        """
        Executes a single study.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        inner_masks : Dict[int, Dict[str, List[int]]]
            Dictionary of inner loops masks, i.e a dictionary with list of idx to use as train, valid and test masks.
        objective : Objective
            The objective.
        """
        self.callbacks.on_study_start(self)

        study = self.search_algorithm.search(
            objective=objective,
            n_trials=self.n_trials,
            masks=inner_masks,
            dataset=dataset,
            callbacks=self.callbacks,
            study_name=f"{self.SPLIT_PREFIX}_{self.outer_loop_state.idx}",
            verbose=self.verbose
        )
        self.study_state.study = study
        self.study_state.best_trial = self._get_best_trial()
        self.callbacks.on_study_end(self)

    def _exec_best_model_evaluation(
            self,
            dataset: ProstateCancerDataset,
            objective: Objective
    ):
        """
        Executes the evaluation of the model using the best trial hyperparameters.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        objective : Objective
            The objective.
        """
        self.callbacks.on_best_model_evaluation_start(self)

        model_evaluation = objective.exec_best_model_evaluation(
            best_trial=self.study_state.best_trial,
            dataset=dataset,
            path_to_save=self.best_model_state.path_to_best_model_folder
        )
        self.best_model_state.score = model_evaluation.score
        self.best_model_state.model = model_evaluation.trained_model
        self.callbacks.on_best_model_evaluation_end(self)

    def _set_seed(self):
        """
        Sets numpy and torch seed.
        """
        if self.seed is not None:
            np_seed(self.seed)
            manual_seed(self.seed)

    def _get_best_trial(self) -> FrozenTrial:
        """
        Retrieves the best trial among all the trials on the pareto front of the current study.

        Returns
        -------
        best_trial : FrozenTrial
            Best trial.
        """
        # TODO : Find a way to choose the best hps set among all the sets on the pareto front. For now, we arbitrarily
        #  choose the first in the list.

        return self.study_state.study.best_trials[0]
