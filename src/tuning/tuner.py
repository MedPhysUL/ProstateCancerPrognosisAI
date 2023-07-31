"""
    @file:              tuner.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     03/2022
    @Last modification: 02/2023

    @Description:      This file is used to define the `Tuner` class in charge of tuning the hyperparameters.
"""

from typing import Dict, List, Optional, Union

import pandas as pd
from monai.utils import set_determinism
from optuna.trial import FrozenTrial

from .callbacks import TuningRecorder
from .callbacks.base import TuningCallback
from .callbacks.containers import TuningCallbackList
from ..data.datasets import Mask, ProstateCancerDataset, Split
from ..models.torch import ModelConfig
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
            recorder: TuningRecorder = None,
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
        recorder : TuningRecorder
            The tuning recorder.
        n_trials : int
            Number of sets of hyperparameters tested.
        seed : Optional[int]
            Random state used for reproducibility.
        verbose : bool
            Whether to print out the trace of the tuner.
        """
        self.n_trials = n_trials
        self.search_algorithm = search_algorithm
        self.seed = seed
        self.verbose = verbose

        self._recorder = recorder
        self._build_callbacks()

        self.best_model_state, self.outer_loop_state, self.study_state, self.tuning_state = None, None, None, None
        self._initialize_states()

    @property
    def callbacks(self) -> Optional[TuningCallbackList]:
        """
        Callbacks to use during the training process.

        Returns
        -------
        callbacks : Optional[TuningCallbackList]
            Callback list
        """
        return self._callbacks

    def _build_callbacks(self):
        """
        Builds the callbacks attribute as a `TuningCallbackList` and sorts the callbacks in this list.
        """
        callbacks: List[TuningCallback] = []

        if self.recorder:
            callbacks += [self.recorder]

        self._callbacks = TuningCallbackList(callbacks)
        self._callbacks.sort()

    @property
    def recorder(self) -> Optional[TuningRecorder]:
        """
        Recorder callback.

        Returns
        -------
        recorder : Optional[TuningRecorder]
            The recorder callback, if there is one.
        """
        return self._recorder

    @recorder.setter
    def recorder(self, recorder: Optional[TuningRecorder]):
        """
        Sets recorder callback.

        Parameters
        ----------
        recorder : Optional[TuningRecorder]
            The recorder callback.
        """
        self._recorder = recorder
        self._build_callbacks()

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
            masks: Dict[int, Dict[str, Union[List[int], Dict[int, Dict[str, List[int]]]]]],
            dataframes: Optional[Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]] = None,
            model_configs: Optional[Dict[int, Dict[str, Union[ModelConfig, Dict[int, ModelConfig]]]]] = None
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
        dataframes : Optional[Dict[int, Dict[str, Union[pd.DataFrame, Dict[int, pd.DataFrame]]]]]
            Dictionary with dataframes to use for different splits.
        model_configs : Optional[Dict[int, Dict[str, Union[ModelConfig, Dict[int, ModelConfig]]]]]
            Dictionary with model configs to use for different splits.
        """
        assert len(dataset.tunable_tasks) > 0, (
            "No tunable task found in the dataset. A tunable task is a task with a `hps_tuning_metric` attribute. "
            "Please add this attribute to the task you want to tune."
        )
        self._initialize_states()
        self._set_seed()

        self.callbacks.on_tuning_start(self)
        self.outer_loop_state.scores = []
        for idx, mask in masks.items():
            self.outer_loop_state.idx = idx
            self._update_dataset(dataset, mask, dataframes[idx][Split.OUTER] if dataframes else None)
            self.outer_loop_state.dataset = dataset

            self.callbacks.on_outer_loop_start(self)
            inner_dataframes = dataframes[idx][Split.INNER] if dataframes else None
            inner_model_configs = model_configs[idx][Split.INNER] if model_configs else None
            self._exec_study(dataset, mask[Mask.INNER], objective, inner_dataframes, inner_model_configs)

            self._update_dataset(dataset, mask, dataframes[idx][Split.OUTER] if dataframes else None)
            self._exec_best_model_evaluation(
                dataset=dataset,
                objective=objective,
                model_configs=model_configs[idx][Split.OUTER] if model_configs else None
            )
            self._exec_inner_loops_best_models_evaluation(
                dataset, mask[Mask.INNER], objective, inner_dataframes, inner_model_configs
            )
            self.callbacks.on_outer_loop_end(self)

        self.callbacks.on_tuning_end(self)

    @staticmethod
    def _update_dataset(
            dataset: ProstateCancerDataset,
            mask: Dict[str, List[int]],
            dataframe: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Updates the dataset with the given mask and dataframe.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset to update.
        mask : Dict[str, List[int]]
            Dictionary of masks.
        dataframe : Optional[pd.DataFrame]
            Dataframe to use for the update.
        """
        if dataframe is not None:
            dataset.update_dataframe(dataframe=dataframe, update_masks=False)

        dataset.update_masks(train_mask=mask[Mask.TRAIN], valid_mask=mask[Mask.VALID], test_mask=mask[Mask.TEST])

    def _exec_study(
            self,
            dataset: ProstateCancerDataset,
            inner_masks: Dict[int, Dict[str, List[int]]],
            objective: Objective,
            inner_dataframes: Optional[Dict[int, pd.DataFrame]] = None,
            inner_model_configs: Optional[Dict[int, Dict[str, ModelConfig]]] = None
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
        inner_dataframes : Optional[Dict[int, pd.DataFrame]]
            Dictionary of dataframes to use for different inner splits.
        inner_model_configs : Optional[Dict[int, Dict[str, ModelConfig]]]
            Dictionary of model configs to use for different inner splits.
        """
        self.callbacks.on_study_start(self)

        study = self.search_algorithm.search(
            objective=objective,
            n_trials=self.n_trials,
            masks=inner_masks,
            dataset=dataset,
            callbacks=self.callbacks,
            dataframes=inner_dataframes,
            model_configs=inner_model_configs,
            study_name=f"{self.SPLIT_PREFIX}_{self.outer_loop_state.idx}",
            verbose=self.verbose
        )
        self.study_state.study = study
        self.study_state.best_trial = self._get_best_trial()
        self.callbacks.on_study_end(self)

    def _exec_inner_loops_best_models_evaluation(
            self,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
            objective: Objective,
            dataframes: Optional[Dict[int, pd.DataFrame]] = None,
            model_configs: Optional[Dict[int, Dict[str, ModelConfig]]] = None
    ):
        """
        Executes the evaluation of the best models of the inner loops.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        masks : Dict[int, Dict[str, List[int]]]
            Dictionary of inner loops masks, i.e a dictionary with list of idx to use as train, valid and test masks.
        objective : Objective
            The objective.
        dataframes : Optional[Dict[int, pd.DataFrame]]
            Dictionary of dataframes to use for different inner splits.
        model_configs : Optional[Dict[int, Dict[str, ModelConfig]]]
            Dictionary of model configs to use for different inner splits.
        """
        if self.recorder.save_inner_splits_best_models:
            for idx, mask in masks.items():
                self._update_dataset(dataset=dataset, mask=mask, dataframe=dataframes[idx] if dataframes else None)

                configs = model_configs[idx] if model_configs else None
                self._exec_best_model_evaluation(dataset, objective, False, idx, configs)

    def _exec_best_model_evaluation(
            self,
            dataset: ProstateCancerDataset,
            objective: Objective,
            outer: bool = True,
            idx: Optional[int] = None,
            model_configs: Optional[Dict[str, ModelConfig]] = None
    ):
        """
        Executes the evaluation of the model using the best trial hyperparameters.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        objective : Objective
            The objective.
        outer : bool
            Whether the evaluation is performed on the outer loop or not.
        idx : Optional[int]
            The index of the INNER loop. None if outer is True.
        model_configs : Optional[Dict[str, ModelConfig]]
            Dictionary of model configs to use for different tasks.
        """
        self.callbacks.on_best_model_evaluation_start(self, outer=outer, idx=idx)

        model_evaluation = objective.exec_best_model_evaluation(
            best_trial=self.study_state.best_trial,
            dataset=dataset,
            model_configs=model_configs,
            path_to_save=self.best_model_state.path_to_current_model_folder,
            seed=self.seed
        )
        self.best_model_state.score = model_evaluation.score
        self.best_model_state.model = model_evaluation.trained_model
        self.callbacks.on_best_model_evaluation_end(self, outer=outer, idx=idx)

    def _set_seed(self):
        """
        Sets numpy and torch seed.
        """
        if self.seed is not None:
            set_determinism(self.seed)

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
