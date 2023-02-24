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

    def init(self, score: ScoreContainer):
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

    def append(self, score: ScoreContainer):
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

    def set_from_history(self, history: HistoryContainer):
        """
        Sets statistics from history.

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
