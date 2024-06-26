"""
    @file:              regression.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `RegressionTask` class.
"""

from typing import Iterable, Optional, Union

from .base import TableTask
from ..losses.single_task.regression import RegressionLoss
from ..metrics.single_task.containers import SingleTaskMetricList
from ..metrics.single_task.regression import RegressionMetric
from ..tools.missing_targets import get_idx_of_nonmissing_regression_targets


class RegressionTask(TableTask):
    """
    A class used to define a Regression task.
    """

    def __init__(
            self,
            target_column: str,
            criterion: Optional[RegressionLoss] = None,
            early_stopping_metric: Optional[RegressionMetric] = None,
            evaluation_metrics: Optional[
                Union[
                    RegressionMetric,
                    Iterable[RegressionMetric],
                    SingleTaskMetricList[RegressionMetric]
                ]
            ] = None,
            hps_tuning_metric: Optional[RegressionMetric] = None,
            name: Optional[str] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[RegressionLoss]
            A loss function.
        early_stopping_metric : Optional[RegressionMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[
                Union[
                    RegressionMetric,
                    Iterable[RegressionMetric],
                    SingleTaskMetricList[RegressionMetric]
                ]
            ]
            A list of metrics to evaluate the trained models.
        hps_tuning_metric : RegressionMetric
            A metric used for Optuna hyperparameters optimization.
        name : Optional[str]
            The name of the task.
        """
        name = name if name else f"{self.__class__.__name__}('target_column'={repr(target_column)})"

        super().__init__(
            hps_tuning_metric=hps_tuning_metric,
            name=name,
            target_column=target_column,
            criterion=criterion,
            early_stopping_metric=early_stopping_metric,
            evaluation_metrics=evaluation_metrics
        )

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_regression_targets

        self._validate_metrics_type(type_=RegressionMetric)
        self._validate_criterion_type(type_=RegressionLoss)

    @property
    def criterion(self) -> Optional[RegressionLoss]:
        return self._criterion

    @property
    def early_stopping_metric(self) -> Optional[RegressionMetric]:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[SingleTaskMetricList[RegressionMetric]]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> Optional[RegressionMetric]:
        return self._hps_tuning_metric

    @property
    def metrics(self) -> SingleTaskMetricList[RegressionMetric]:
        return super().metrics

    @property
    def unique_metrics(self) -> SingleTaskMetricList[RegressionMetric]:
        return super().unique_metrics
