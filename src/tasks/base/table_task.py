"""
    @file:              table_task.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `TableTask` class, i.e. a task to do on table data.
"""

from abc import ABC
from typing import Iterable, Optional, Union

from ...losses.single_task.binary_classification import BinaryClassificationLoss
from ...losses.single_task.regression import RegressionLoss
from ...losses.single_task.survival_analysis import SurvivalAnalysisLoss
from ...metrics.single_task.binary_classification import BinaryClassificationMetric
from ...metrics.single_task.containers import SingleTaskMetricList
from ...metrics.single_task.regression import RegressionMetric
from ...metrics.single_task.survival_analysis import SurvivalAnalysisMetric
from .task import Task


class TableTask(Task, ABC):
    """
    An abstract class representing a task to do on table data.
    """

    def __init__(
            self,
            name: str,
            target_column: str,
            criterion: Optional[Union[BinaryClassificationLoss, RegressionLoss, SurvivalAnalysisLoss]] = None,
            early_stopping_metric: Optional[Union[
                BinaryClassificationMetric,
                RegressionMetric,
                SurvivalAnalysisMetric
            ]] = None,
            evaluation_metrics: Optional[Union[
                Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric],
                Iterable[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]],
                SingleTaskMetricList[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]]
            ]] = None,
            hps_tuning_metric: Optional[Union[
                BinaryClassificationMetric,
                RegressionMetric,
                SurvivalAnalysisMetric
            ]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : str
            The name of the task.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[Union[BinaryClassificationLoss, RegressionLoss, SurvivalAnalysisLoss]]
            A loss function.
        early_stopping_metric : Optional[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]]
            A metric used for early stopping.
        evaluation_metrics : Optional[
                Union[
                    Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric],
                    Iterable[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]],
                    SingleTaskMetricList[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]]
                ]
            ]
            A list of metrics to evaluate the trained models.
        hps_tuning_metric : Optional[Union[BinaryClassificationMetric, RegressionMetric, SurvivalAnalysisMetric]]
            A metric used for Optuna hyperparameters optimization.
        """
        super().__init__(
            hps_tuning_metric=hps_tuning_metric,
            name=name,
            criterion=criterion,
            early_stopping_metric=early_stopping_metric,
            evaluation_metrics=evaluation_metrics
        )
        self._target_column = target_column

    @property
    def target_column(self) -> str:
        """
        Target column.

        Returns
        -------
        target_column : str
            Name of the column containing the targets associated to this task.
        """
        return self._target_column
