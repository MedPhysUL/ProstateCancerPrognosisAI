"""
    @file:              states.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define a few states used within the `Tuner` class.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from optuna.study import Study
from optuna.trial import FrozenTrial

from ..data.datasets import ProstateCancerDataset
from ..models.base.model import Model
from .objectives.base.containers import HistoryContainer, ScoreContainer, StatisticsContainer


@dataclass
class BestModelState:
    """
    This class is used to store the current best model state.

    Elements
    --------
    path_to_best_models_folder: str
        The path to the current folder containing the best models records.
    path_to_current_model_folder: str
        The path to the current folder containing the current model records.
    score : ScoreContainer
        The current best model score.
    model : Model
        Best model.
    """
    path_to_best_models_folder: str = None
    path_to_current_model_folder: str = None
    score: ScoreContainer = None
    model: Model = None


@dataclass
class OuterLoopState:
    """
    This class is used to store the current outer loop state.

    Elements
    --------
    dataset : ProstateCancerDataset
        The current dataset.
    idx : int
        The index of the current outer loop.
    path_to_outer_loop_folder: str
        The path to the current folder containing outer loops records.
    """
    dataset: ProstateCancerDataset = None
    idx: int = None
    path_to_outer_loop_folder: str = None


@dataclass
class StudyState:
    """
    This class is used to store the current study state.

    Elements
    --------
    best_trial : FrozenTrial
        The best trial.
    study : Study
        The current study.
    """
    best_trial: FrozenTrial = None
    study: Study = None


@dataclass
class TuningState:

    INFO = "info"
    MEAN = "mean"
    STD = "std"

    """
    This class is used to store the current tuning state.

    Elements
    --------
    hyperparameters : List[Dict[str, Any]]
        List of hyperparameters.
    hyperparameters_importance : List[Dict[str, Dict[str, Any]]]
        List of hyperparameters importance.
    scores : List[ScoreContainer]
        List of scores.
    """
    hyperparameters: List[Dict[str, Any]] = field(default_factory=list)
    hyperparameters_importance: List[Dict[str, Dict[str, Any]]] = field(default_factory=list)
    scores: List[ScoreContainer] = field(default_factory=list)

    @property
    def hyperparameters_history(self) -> Dict[str, Any]:
        """
        Tuning scores history.

        Returns
        -------
        history : Dict[str, Any]
            Hyperparameters history container.
        """
        history = {}
        for params in self.hyperparameters:
            for hp_name, hp_value in params.items():
                if hp_name in history.keys():
                    history[hp_name].append(hp_value)
                else:
                    history[hp_name] = [hp_value]

        return history

    @property
    def hyperparameters_statistics(self) -> Dict[str, Any]:
        """
        Tuning scores statistics (mean, std, etc.).

        Returns
        -------
        statistics : Dict[str, Any]
            Hyperparameters statistics container.
        """
        statistics = {}
        history = self.hyperparameters_history
        for hp_name, hp_values in history.items():
            statistics[hp_name] = {}
            if all(isinstance(value, int) or isinstance(value, float) for value in hp_values):
                statistics[hp_name][self.MEAN] = float(np.mean(hp_values))
                statistics[hp_name][self.STD] = float(np.std(hp_values))
            else:
                statistics[hp_name][self.INFO] = {x: hp_values.count(x) for x in hp_values}

        return statistics

    @property
    def hyperparameters_importance_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Tuning scores history.

        Returns
        -------
        history : Dict[str, Dict[str, Any]]
            Hyperparameters history container.
        """
        history = {}
        for splits in self.hyperparameters_importance:
            for task, hps_importance in splits.items():
                if task not in history.keys():
                    history[task] = {}
                for hp_name, hp_value in hps_importance.items():
                    if hp_name in history[task].keys():
                        history[task][hp_name].append(hp_value)
                    else:
                        history[task][hp_name] = [hp_value]

        return history

    @property
    def hyperparameters_importance_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Tuning scores statistics (mean, std, etc.).

        Returns
        -------
        statistics : Dict[str, Dict[str, Any]]
            Hyperparameters statistics container.
        """
        statistics = {}
        history = self.hyperparameters_importance_history
        for task, hps_importance in history.items():
            statistics[task] = {}
            for hp_name, hp_values in hps_importance.items():
                statistics[task][hp_name] = {}
                statistics[task][hp_name][self.MEAN] = float(np.mean(hp_values))
                statistics[task][hp_name][self.STD] = float(np.std(hp_values))

        return statistics

    @property
    def scores_history(self) -> HistoryContainer:
        """
        Tuning scores history.

        Returns
        -------
        history_container : HistoryContainer
            Trial history container.
        """
        history = HistoryContainer()
        history.build(self.scores)

        return history

    @property
    def scores_statistics(self) -> StatisticsContainer:
        """
        Tuning scores statistics (mean, std, etc.).

        Returns
        -------
        statistics_container : StatisticsContainer
            Statistics container.
        """
        statistics = StatisticsContainer()
        statistics.build(self.scores_history)

        return statistics
