"""
    @file:              tasks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the different possible tasks.
"""

from abc import ABC
from copy import copy
from typing import Any, Dict, Iterable, Optional, Type, Union

from src.utils.metric_list import MetricList
from src.utils.metrics import BinaryClassificationMetric, Metric, RegressionMetric, SegmentationMetric
from src.utils.losses import BinaryClassificationLoss, Loss, RegressionLoss, SegmentationLoss


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


class TableTask(Task):
    """
    An abstract class representing a task to do on table data.
    """

    def __init__(
            self,
            hps_tuning_metric: Metric,
            name: str,
            target_column: str,
            criterion: Optional[Union[BinaryClassificationLoss, RegressionLoss]] = None,
            early_stopping_metric: Optional[Metric] = None,
            evaluation_metrics: Optional[Union[Metric, Iterable[Metric], MetricList]] = None,

    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        hps_tuning_metric : Metric
            A metric used for Optuna hyperparameters optimization.
        name : str
            The name of the task.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[Union[BinaryClassificationLoss, RegressionLoss]]
            A loss function.
        early_stopping_metric : Optional[Metric]
            A metric used for early stopping.
        evaluation_metrics : Optional[Union[Metric, Iterable[Metric], MetricList]]
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


class ClassificationTask(TableTask):
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
                Union[BinaryClassificationMetric, Iterable[BinaryClassificationMetric], MetricList]
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
                Union[BinaryClassificationMetric, Iterable[BinaryClassificationMetric], MetricList]
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


class RegressionTask(TableTask):
    """
    A class used to define a Regression task.
    """

    def __init__(
            self,
            hps_tuning_metric: RegressionMetric,
            target_column: str,
            criterion: Optional[RegressionLoss] = None,
            early_stopping_metric: Optional[RegressionMetric] = None,
            evaluation_metrics: Optional[Union[RegressionMetric, Iterable[RegressionMetric], MetricList]] = None,
            name: Optional[str] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        hps_tuning_metric : RegressionMetric
            A metric used for Optuna hyperparameters optimization.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[RegressionLoss]
            A loss function.
        early_stopping_metric : Optional[RegressionMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[Union[RegressionMetric, Iterable[RegressionMetric], MetricList]]
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

        self._validate_metrics_type(type_=RegressionMetric)
        self._validate_criterion_type(type_=RegressionLoss)

    @property
    def criterion(self) -> Optional[RegressionLoss]:
        return self._criterion

    @property
    def early_stopping_metric(self) -> Optional[RegressionMetric]:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[MetricList[RegressionMetric]]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> RegressionMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> MetricList[BinaryClassificationMetric]:
        return super().metrics

    @property
    def unique_metrics(self) -> MetricList[BinaryClassificationMetric]:
        return super().unique_metrics


class SegmentationTask(Task):
    """
    A class used to define a Segmentation task.
    """

    def __init__(
            self,
            criterion: SegmentationLoss,
            hps_tuning_metric: SegmentationMetric,
            modality: str,
            organ: str,
            early_stopping_metric: Optional[SegmentationMetric] = None,
            evaluation_metrics: Optional[Union[SegmentationMetric, Iterable[SegmentationMetric], MetricList]] = None,
            name: Optional[str] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        criterion : SegmentationLoss
            A loss function.
        hps_tuning_metric : SegmentationMetric
            A metric used for Optuna hyperparameters optimization.
        modality : str
            Modality on which segmentation was performed.
        organ : str
            Segmented organ.
        early_stopping_metric : Optional[SegmentationMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[Union[SegmentationMetric, Iterable[SegmentationMetric], MetricList]]
            A list of metrics to evaluate the trained models.
        name : Optional[str]
            The name of the task.
        """
        default_name = f"{self.__class__.__name__}('modality'={repr(modality)}, 'organ'={repr(organ)})"
        name = name if name else default_name

        super().__init__(
            hps_tuning_metric=hps_tuning_metric,
            name=name,
            criterion=criterion,
            early_stopping_metric=early_stopping_metric,
            evaluation_metrics=evaluation_metrics
        )

        self._validate_metrics_type(type_=SegmentationMetric)
        self._validate_criterion_type(type_=SegmentationLoss)

        self.modality = modality
        self.organ = organ

    @property
    def criterion(self) -> Optional[SegmentationLoss]:
        return self._criterion

    @property
    def early_stopping_metric(self) -> Optional[SegmentationMetric]:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[MetricList[SegmentationMetric]]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> SegmentationMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> MetricList[SegmentationMetric]:
        return super().metrics

    @property
    def unique_metrics(self) -> MetricList[SegmentationMetric]:
        return super().unique_metrics
