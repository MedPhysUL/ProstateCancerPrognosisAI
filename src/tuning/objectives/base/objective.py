"""
    @file:              objective.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Objective` class used for hyperparameters tuning.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from optuna.trial import FrozenTrial, Trial

from ...callbacks.containers import TuningCallbackList
from .containers import ModelEvaluationContainer, ScoreContainer
from ....data.datasets import Mask, ProstateCancerDataset
from ...hyperparameters.containers import HyperparameterDict
from ....models.base.model import Model
from ....models.torch import ModelConfig
from .states import InnerLoopState, TrialState


class Objective(ABC):
    """
    Abstract objective class to use within the tuner.
    """

    DATASET_KEY = "dataset"
    MODEL_INSTANCE_KEY = "model"
    CONFIGS_KEY = "configs"
    SEED_KEY = "seed"
    TRAIN_METHOD_PARAMS_KEY = "train"

    def __init__(self) -> None:
        """
        Sets protected and public attributes of the objective.
        """
        self.inner_loop_state = InnerLoopState()
        self.trial_state = TrialState()

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

    def __call__(
            self,
            trial: Trial,
            callbacks: TuningCallbackList,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
            dataframes: Optional[Dict[int, pd.DataFrame]] = None,
            model_configs: Optional[Dict[int, Dict[str, ModelConfig]]] = None
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
        dataframes : Optional[Dict[int, pd.DataFrame]]
            Dictionary of dataframes to use for different inner splits.
        model_configs : Optional[Dict[int, Dict[str, ModelConfig]]]
            Dictionary of model configs to use for different inner splits.

        Returns
        -------
        scores : List[float]
            List of task scores associated with the set of hyperparameters.
        """
        self.trial_state.trial = trial
        callbacks.on_trial_start(self)

        suggestion = self.hyperparameters.suggest(trial)

        scores = []
        for idx, mask in enumerate(masks.values()):
            self._update_dataset(dataset=dataset, mask=mask, dataframe=dataframes[idx] if dataframes else None)
            self.inner_loop_state.dataset = dataset
            self.inner_loop_state.idx = idx

            self._exec_inner_loop = self._build_inner_loop_runner()

            if model_configs:
                suggestion[self.TRAIN_METHOD_PARAMS_KEY][self.MODEL_INSTANCE_KEY][self.CONFIGS_KEY] = model_configs[idx]

            score = self._exec_inner_loop(callbacks=callbacks, suggestion=suggestion)
            scores.append(score)

        self.trial_state.scores = scores

        tuning_metric_test_scores = [
            self.trial_state.statistics.test[task.name][task.hps_tuning_metric.name].mean
            for task in dataset.tunable_tasks
        ]

        callbacks.on_trial_end(self)

        return tuning_metric_test_scores

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

    def exec_best_model_evaluation(
            self,
            best_trial: FrozenTrial,
            dataset: ProstateCancerDataset,
            path_to_save: str,
            seed: int,
            model_configs: Optional[Dict[str, ModelConfig]] = None,
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
        seed : int
            Seed to use.
        model_configs : Optional[Dict[str, ModelConfig]]
            Model configs to use.

        Returns
        -------
        model_evaluation : ModelEvaluationContainer
            Model evaluation.
        """
        past_suggestion = self.hyperparameters.retrieve_past_suggestion(best_trial)
        past_suggestion[self.TRAIN_METHOD_PARAMS_KEY][self.MODEL_INSTANCE_KEY][self.SEED_KEY] = seed

        if model_configs:
            past_suggestion[self.TRAIN_METHOD_PARAMS_KEY][self.MODEL_INSTANCE_KEY][self.CONFIGS_KEY] = model_configs

        hyperparameters = self.hyperparameters.build(past_suggestion)
        model_evaluation = self._test_hyperparameters(
            dataset=dataset,
            hyperparameters=hyperparameters,
            path_to_save=path_to_save
        )

        return model_evaluation

    def _build_inner_loop_runner(self) -> Callable:
        """
        Builds the function run in parallel for each set of hyperparameters and return the score.

        Returns
        -------
        run_inner_loop : Callable
            Function that train a single model using given masks and trial.
        """

        def exec_inner_loop(
                callbacks: TuningCallbackList,
                suggestion: Dict[str, Any]
        ) -> ScoreContainer:
            """
            Trains a single model using given masks and hyperparameters.

            Parameters
            ----------
            callbacks : TuningCallbackList
                Callbacks to use during tuning.
            suggestion : Dict[str, Any]
                Suggested hyperparameters for this trial.

            Returns
            -------
            score : ScoreContainer
                Score values.
            """
            callbacks.on_inner_loop_start(self)
            hyperparameters = self.hyperparameters.build(suggestion)
            model_evaluation = self._test_hyperparameters(
                dataset=self.inner_loop_state.dataset,
                hyperparameters=hyperparameters,
                path_to_save=self.inner_loop_state.path_to_inner_loop_folder
            )
            score = model_evaluation.score
            self.inner_loop_state.score = score
            callbacks.on_inner_loop_end(self)

            return score

        return exec_inner_loop

    @staticmethod
    def _get_model_evaluation(
            trained_model: Model,
            dataset: ProstateCancerDataset
    ) -> ModelEvaluationContainer:
        """
        Gets model evaluation given a trained model and a dataset.

        Parameters
        ----------
        trained_model : Model
            A trained model.
        dataset : ProstateCancerDataset
            A dataset.

        Returns
        -------
        model_evaluation : ModelEvaluationContainer
            Model evaluation.
        """
        # We find the optimal threshold for each classification tasks
        trained_model.fix_thresholds_to_optimal_values(dataset)

        # We calculate the scores on the different tasks on the different sets
        train_set_scores = trained_model.compute_score_on_dataset(dataset, dataset.train_mask)
        valid_set_scores = trained_model.compute_score_on_dataset(dataset, dataset.valid_mask)
        test_set_scores = trained_model.compute_score_on_dataset(dataset, dataset.test_mask)
        score = ScoreContainer(train=train_set_scores, valid=valid_set_scores, test=test_set_scores)

        return ModelEvaluationContainer(trained_model=trained_model, score=score)

    @abstractmethod
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
        raise NotImplementedError
