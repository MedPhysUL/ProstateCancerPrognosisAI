"""
    @file:              containers.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define a few containers used in the objective class.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ....models.base.model import Model


@dataclass
class ScoreContainer:
    """
    This class is used to store scores.

    Elements
    --------
    train : Dict[str, Dict[str, float]]
        Training set scores.
    valid : Dict[str, Dict[str, float]]
        Valid set scores.
    test : Dict[str, Dict[str, float]]
        Test set scores.
    """
    train: Dict[str, Dict[str, float]] = field(default_factory=dict)
    valid: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class HistoryContainer:
    """
    This class is used to store scores' history.

    Elements
    --------
    train : Dict[str, Dict[str, List[float]]]
        Training set scores' history.
    valid : Dict[str, Dict[str, List[float]]]
        Valid set scores' history.
    test : Dict[str, Dict[str, List[float]]]
        Test set scores' history.
    """
    train: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    valid: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    test: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def build(self, scores: List[ScoreContainer]):
        """
        Builds history container from list of score.

        Parameters
        ----------
        scores : List[ScoreContainer]
            A list of score container.
        """
        for score_idx, score in enumerate(scores):
            if score_idx == 0:
                self._init(score)
            else:
                self._append(score)

    def _init(self, score: ScoreContainer):
        """
        Initializes history using a single score.

        Parameters
        ----------
        score : ScoreContainer
            Score container.
        """
        for k, v in vars(score).items():
            if k in vars(self):
                vars(self)[k] = {}
                for task_name, metrics in v.items():
                    vars(self)[k][task_name] = {}
                    for metric_name, value in metrics.items():
                        vars(self)[k][task_name][metric_name] = [value]

    def _append(self, score: ScoreContainer):
        """
        Appends a single score to the history.

        Parameters
        ----------
        score : ScoreContainer
            Score container.
        """
        for k, v in vars(score).items():
            if k in vars(self):
                for task_name, metrics in v.items():
                    for metric_name, value in metrics.items():
                        vars(self)[k][task_name][metric_name].append(value)


@dataclass
class ScoreStatisticsContainer:
    """
    This class is used to store scores' statistics.

    Elements
    --------
    mean : float
        Mean of the scores in the history.
    std : float
        Standard deviation (std) of the scores in the history.
    """
    mean: float = None
    std: float = None


@dataclass
class StatisticsContainer:
    """
    This class is used to store scores' history.

    Elements
    --------
    train : Dict[str, Dict[str, ScoreStatisticsContainer]]
        Training set scores' statistics.
    valid : Dict[str, Dict[str, ScoreStatisticsContainer]]
        Valid set scores' statistics.
    test : Dict[str, Dict[str, ScoreStatisticsContainer]]
        Test set scores' statistics.
    """
    train: Dict[str, Dict[str, ScoreStatisticsContainer]] = field(default_factory=dict)
    valid: Dict[str, Dict[str, ScoreStatisticsContainer]] = field(default_factory=dict)
    test: Dict[str, Dict[str, ScoreStatisticsContainer]] = field(default_factory=dict)

    def build(self, history: HistoryContainer):
        """
        Builds statistics container from history.

        Parameters
        ----------
        history : HistoryContainer
            History container.
        """
        for k, v in vars(history).items():
            if k in vars(self):
                vars(self)[k] = {}
                for task_name, metrics in v.items():
                    vars(self)[k][task_name] = {}
                    for metric_name, value in metrics.items():
                        vars(self)[k][task_name][metric_name] = ScoreStatisticsContainer(
                            mean=float(np.mean(value)),
                            std=float(np.std(value))
                        )


@dataclass
class ModelEvaluationContainer:
    """
    This class is used to store model evaluation.

    Elements
    --------
    trained_model : Model
        Model state.
    score : ScoreContainer
        Model score.
    """
    trained_model: Model = None
    score: ScoreContainer = None
