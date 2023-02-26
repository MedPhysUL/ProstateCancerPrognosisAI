"""
    @file:              states.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define a few states used within the `Objective` class.
"""

from dataclasses import dataclass, field
from typing import List

from optuna.trial import Trial

from ...callbacks.containers import TuningCallbackList
from .containers import HistoryContainer, ScoreContainer, StatisticsContainer
from ....data.datasets import ProstateCancerDataset


@dataclass
class InnerLoopState:
    """
    This class is used to store the current inner loop state. It is extremely useful for the callbacks to access the
    current inner loop state and to modify the tuning process.

    Elements
    --------
    callbacks : TuningCallbackList
        Callbacks to use during tuning.
    dataset : ProstateCancerDataset
        The current dataset.
    idx : int
        The index of the current inner loop.
    path_to_inner_loop_folder: str
        The path to the current folder containing inner loops records.
    score : ScoreContainer
        The current inner loop score.
    """
    callbacks: TuningCallbackList = None
    dataset: ProstateCancerDataset = None
    idx: int = None
    path_to_inner_loop_folder: str = None
    score: ScoreContainer = None


@dataclass
class TrialState:
    """
    This class is used to store the current trial state. It is extremely useful for the callbacks to access the current
    trial state and to modify the tuning process.

    Elements
    --------
    path_to_trial_folder : str
        The path to the folder containing trials records.
    scores : List[ScoreContainer]
        The current inner loops scores.
    trial : Trial
        The current trial.
    """
    path_to_trial_folder: str = None
    trial: Trial = None
    scores: List[ScoreContainer] = field(default_factory=list)

    @property
    def history(self) -> HistoryContainer:
        """
        Trial history.

        Returns
        -------
        history_container : HistoryContainer
            Trial history container.
        """
        history = HistoryContainer()
        history.build(self.scores)

        return history

    @property
    def statistics(self) -> StatisticsContainer:
        """
        Trial statistics (mean, std, etc.)

        Returns
        -------
        statistics_container : StatisticsContainer
            Statistics container.
        """
        statistics = StatisticsContainer()
        statistics.build(self.history)

        return statistics
