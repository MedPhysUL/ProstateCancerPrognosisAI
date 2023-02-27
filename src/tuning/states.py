"""
    @file:              states.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define a few states used within the `Tuner` class.
"""

from dataclasses import dataclass

from optuna.study import Study
from optuna.trial import FrozenTrial

from ..data.datasets import ProstateCancerDataset
from ..models.base.model import Model
from .objectives.base import ScoreContainer


@dataclass
class BestModelState:
    """
    This class is used to store the current best model state.

    Elements
    --------
    path_to_best_model_folder: str
        The path to the current folder containing the best model records.
    score : ScoreContainer
        The current best model score.
    model : Model
        Best model.
    """
    path_to_best_model_folder: str = None
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
    study : Study
        The current study.
    best_trial : FrozenTrial
        The best trial.
    """
    study: Study = None
    best_trial: FrozenTrial = None
