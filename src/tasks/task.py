"""
    @file:              task.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `Task` class.
"""

from abc import ABC
from copy import copy
from typing import Any, Dict, Iterable, Optional, Type, Union

from ..losses.loss import Loss
from ..metrics.metric import Metric
from ..metrics.metric_list import MetricList


class Task(ABC):
    """
    An abstract class representing a task.
    """

    PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES = ["_hps_tuning_metric", "_criterion", "_early_stopping_metric"]
    TORCH_LIKE_SERIALIZABLE_ATTRIBUTES = ["_evaluation_metrics"]

    def __init__(
            self,
            hps_tuning_metric: Metric,
            criterion: Optional[Loss] = None,
            early_stopping_metric: Optional[Metric] = None,
            evaluation_metrics: Optional[Union[Metric, Iterable[Metric], MetricList]] = None,
            name: str = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        hps_tuning_metric : Metric
            A metric used for Optuna hyperparameters optimization.
        criterion : Optional[Loss]
            A loss function.
        early_stopping_metric : Optional[Metric]
            A metric used for early stopping.
        evaluation_metrics : Optional[Union[Metric, Iterable[Metric], MetricList]]
            A list of metrics to evaluate the trained models on.
        name : str
            The name of the task.
        """
        self._name = name

        self._criterion = criterion
        self._early_stopping_metric = early_stopping_metric
        self._evaluation_metrics = MetricList(evaluation_metrics)
        self._hps_tuning_metric = hps_tuning_metric

    @property
    def criterion(self) -> Optional[Loss]:
        """
        Criterion.

        Returns
        -------
        criterion : Optional[Loss]
            A loss function.
        """
        return self._criterion

    @property
    def early_stopping_metric(self) -> Optional[Metric]:
        """
        Early stopping metric.

        Returns
        -------
        early_stopping_metric : Optional[Metric]
            A metric used for early stopping.
        """
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[MetricList]:
        """
        Evaluation metrics.

        Returns
        -------
        evaluation_metrics : Optional[MetricList]
            A list of metrics to evaluate the trained models on.
        """
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> Metric:
        """
        Hyperparameters tuning metric.

        Returns
        -------
        hps_tuning_metric : Metric
            A metric used for Optuna hyperparameters optimization.
        """
        return self._hps_tuning_metric

    @property
    def metrics(self) -> MetricList:
        """
        'MetricList' containing all given metrics.

        Returns
        -------
        metric_list : MetricList
            A metric list containing all metrics.
        """
        metrics = [self.hps_tuning_metric]
        if self.early_stopping_metric:
            metrics.append(self.early_stopping_metric)
        if self.evaluation_metrics:
            metrics.extend(self.evaluation_metrics)

        return MetricList(metrics)

    @property
    def name(self) -> str:
        """
        Task name

        Returns
        -------
        name : str
            Task name.
        """
        return self._name

    @property
    def unique_metrics(self) -> MetricList:
        """
        'MetricList' containing only the unique metrics. Metrics with the same name are considered identical.

        Returns
        -------
        metric_list : MetricList
            A metric list containing only the unique metrics.
        """
        unique_metrics, unique_names = [], []
        for metric in self.metrics:
            if metric.name not in unique_names:
                unique_metrics.append(metric)
                unique_names.append(metric.name)

        return MetricList(unique_metrics)

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state of the task.

        Returns
        -------
        state: Dict[str, Any]
            The state of the task.
        """
        state = {}

        for k, v in vars(self).items():
            if k in self.TORCH_LIKE_SERIALIZABLE_ATTRIBUTES:
                if v:
                    state[k] = v.state_dict()
                else:
                    state[k] = None
            elif k in self.PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES:
                if v:
                    state[k] = vars(v).copy()
                else:
                    state[k] = None
            else:
                state[k] = copy(v)

        return state

    def _validate_criterion_type(self, type_: Type):
        """
        Validate criterion type. Raise an AssertionError if the criterion is not of the required type.

        Parameters
        ----------
        type_ : Type
            Required criterion's type.
        """
        if self._criterion and not isinstance(self._criterion, type_):
            raise AssertionError(
                f"The 'criterion' of a '{self.__class__.__name__}' should be of type '{type_.__name__}'."
            )

    def _validate_metrics_type(self, type_: Type):
        """
        Validate all metrics type. Raise an AssertionError if any of the metric is not of the required type.

        Parameters
        ----------
        type_ : Type
            Required metrics' type.
        """
        if self.metrics and not all(isinstance(m, type_) for m in self.metrics):
            raise AssertionError(
                f"All metrics of a '{self.__class__.__name__}' should be of type '{type_.__name__}'."
            )
