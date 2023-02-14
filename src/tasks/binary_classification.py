"""
    @file:              binary_classification.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `BinaryClassificationTask` class.
"""

from typing import Iterable, Optional, Union

from ..losses.binary_classification import BinaryClassificationLoss
from ..metrics.binary_classification import BinaryClassificationMetric
from ..metrics.metric_list import MetricList
from .table import TableTask


class BinaryClassificationTask(TableTask):
    """
    A class used to define a Classification task.
    """

    PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES = TableTask.PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES + ["_decision_threshold_metric"]

    def __init__(
            self,
            decision_threshold_metric: BinaryClassificationMetric,
            hps_tuning_metric: BinaryClassificationMetric,
            target_column: str,
            criterion: Optional[BinaryClassificationLoss] = None,
            early_stopping_metric: Optional[BinaryClassificationMetric] = None,
            evaluation_metrics: Optional[
                Union[
                    BinaryClassificationMetric,
                    Iterable[BinaryClassificationMetric],
                    MetricList[BinaryClassificationMetric]
                ]
            ] = None,
            name: Optional[str] = None,
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        decision_threshold_metric : BinaryClassificationMetric
            A metric whose optimized threshold is used to make class predictions from probability predictions.
        hps_tuning_metric : BinaryClassificationMetric
            A metric used for Optuna hyperparameters optimization.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[BinaryClassificationLoss]
            A loss function.
        early_stopping_metric : Optional[BinaryClassificationMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[
                Union[
                    BinaryClassificationMetric,
                    Iterable[BinaryClassificationMetric],
                    MetricList[BinaryClassificationMetric]
                ]
            ]
            A list of metrics to evaluate the trained models.
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

        self._decision_threshold_metric = decision_threshold_metric

        self._validate_metrics_type(type_=BinaryClassificationMetric)
        self._validate_criterion_type(type_=BinaryClassificationLoss)

    @property
    def criterion(self) -> Optional[BinaryClassificationLoss]:
        return self._criterion

    @property
    def decision_threshold_metric(self) -> Optional[BinaryClassificationMetric]:
        return self._decision_threshold_metric

    @property
    def early_stopping_metric(self) -> Optional[BinaryClassificationMetric]:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[MetricList[BinaryClassificationMetric]]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> BinaryClassificationMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> MetricList[BinaryClassificationMetric]:
        metrics = super().metrics
        metrics.append(self.decision_threshold_metric)
        return MetricList(metrics)

    @property
    def unique_metrics(self) -> MetricList[BinaryClassificationMetric]:
        unique_metrics = super().unique_metrics
        unique_names = [metric.name for metric in unique_metrics]

        if self.decision_threshold_metric.name not in unique_names:
            unique_metrics.append(self.decision_threshold_metric)
            unique_names.append(self.decision_threshold_metric.name)

        return MetricList(unique_metrics)
