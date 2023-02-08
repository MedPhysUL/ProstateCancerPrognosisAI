"""
    @file:              table_task.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `TableTask` class, i.e. a task to do on table data.
"""

from abc import ABC
from typing import Iterable, Optional, Union

from src.losses.binary_classification import BinaryClassificationLoss
from src.losses.regression import RegressionLoss
from src.metrics.binary_classification import BinaryClassificationMetric
from src.metrics.regression import RegressionMetric
from src.metrics.metric_list import MetricList
from src.tasks.task import Task


class TableTask(Task, ABC):
    """
    An abstract class representing a task to do on table data.
    """

    def __init__(
            self,
            hps_tuning_metric: Union[BinaryClassificationMetric, RegressionMetric],
            name: str,
            target_column: str,
            criterion: Optional[Union[BinaryClassificationLoss, RegressionLoss]] = None,
            early_stopping_metric: Optional[Union[BinaryClassificationMetric, RegressionMetric]] = None,
            evaluation_metrics: Optional[
                Union[
                    Union[BinaryClassificationMetric, RegressionMetric],
                    Iterable[Union[BinaryClassificationMetric, RegressionMetric]],
                    MetricList[Union[BinaryClassificationMetric, RegressionMetric]]
                ]
            ] = None,

    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        hps_tuning_metric : Union[BinaryClassificationMetric, RegressionMetric]
            A metric used for Optuna hyperparameters optimization.
        name : str
            The name of the task.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[Union[BinaryClassificationLoss, RegressionLoss]]
            A loss function.
        early_stopping_metric : Optional[Union[BinaryClassificationMetric, RegressionMetric]]
            A metric used for early stopping.
        evaluation_metrics : Optional[
                Union[
                    Union[BinaryClassificationMetric, RegressionMetric],
                    Iterable[Union[BinaryClassificationMetric, RegressionMetric]],
                    MetricList[Union[BinaryClassificationMetric, RegressionMetric]]
                ]
            ]
            A list of metrics to evaluate the trained models.
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
