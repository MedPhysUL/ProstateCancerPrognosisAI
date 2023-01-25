"""
    @file:              tasks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 01/2023

    @Description:       This file is used to define the different possible tasks.
"""

from abc import ABC
from typing import List, Optional, Type, Union

from src.utils.score_metrics import BinaryClassificationMetric, Metric, RegressionMetric, SegmentationMetric
from src.utils.losses import BinaryClassificationLoss, Loss, RegressionLoss, SegmentationLoss


class Task(ABC):
    """
    An abstract class representing a task.
    """

    _instance_names = []

    def __init__(
            self,
            hps_tuning_metric: Metric,
            criterion: Optional[Loss] = None,
            early_stopping_metric: Optional[Metric] = None,
            evaluation_metrics: Optional[List[Metric]] = None,
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
        evaluation_metrics : Optional[List[Metric]]
            A list of metrics to evaluate the trained models on.
        name : str
            The name of the task.
        """
        assert name not in self._instance_names, "Tasks name must be unique."
        self._instance_names.append(name)
        self._name = name

        self._criterion = criterion

        self._set_hps_tuning_metric(hps_tuning_metric)
        self._set_early_stopping_metric(early_stopping_metric)
        self._set_evaluation_metrics(evaluation_metrics)

    @property
    def criterion(self) -> Optional[Loss]:
        return self._criterion

    @property
    def early_stopping_metric(self) -> Metric:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> List[Metric]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> Metric:
        return self._hps_tuning_metric

    @property
    def metrics(self) -> List[Metric]:
        return self._metrics

    @property
    def name(self) -> str:
        return self._name

    def _set_hps_tuning_metric(self, hps_tuning_metric: Metric):
        self._hps_tuning_metric = hps_tuning_metric
        self._unique_metric_names = {hps_tuning_metric.name}
        self._metrics = [hps_tuning_metric]

    def _set_early_stopping_metric(self, early_stopping_metric: Metric):
        if early_stopping_metric:
            self._early_stopping_metric = early_stopping_metric
            if early_stopping_metric.name not in self._unique_metric_names:
                self._unique_metric_names.add(early_stopping_metric.name)
                self._metrics.append(early_stopping_metric)
        else:
            self._early_stopping_metric = self._hps_tuning_metric

    def _set_evaluation_metrics(self, evaluation_metrics: List[Metric]):
        if evaluation_metrics:
            self._evaluation_metrics = evaluation_metrics
            for m in evaluation_metrics:
                if m.name not in self._unique_metric_names:
                    self._unique_metric_names.add(m.name)
                    self._metrics.append(m)
        else:
            self._evaluation_metrics = []

    def _validate_metrics_type(self, type_: Type):
        if self.metrics and not all(isinstance(m, type_) for m in self.metrics):
            raise AssertionError(
                f"All metrics of a '{self.__class__.__name__}' should be of type '{type_.__name__}'."
            )

    def _validate_criterion_type(self, type_: Type):
        if self._criterion and not isinstance(self._criterion, type_):
            raise AssertionError(
                f"The 'criterion' of a '{self.__class__.__name__}' should be of type '{type_.__name__}'."
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
            evaluation_metrics: Optional[List[Metric]] = None,

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
        evaluation_metrics : Optional[List[Metric]]
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
        return self._target_column


class ClassificationTask(TableTask):
    """
    A class used to define a Classification task.
    """

    def __init__(
            self,
            hps_tuning_metric: BinaryClassificationMetric,
            target_column: str,
            criterion: Optional[BinaryClassificationLoss] = None,
            decision_threshold_metric: Optional[BinaryClassificationMetric] = None,
            early_stopping_metric: Optional[BinaryClassificationMetric] = None,
            evaluation_metrics: Optional[List[BinaryClassificationMetric]] = None,
            name: Optional[str] = None,
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        hps_tuning_metric : BinaryClassificationMetric
            A metric used for Optuna hyperparameters optimization.
        target_column : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[BinaryClassificationLoss]
            A loss function.
        decision_threshold_metric : Optional[BinaryClassificationMetric]
            A metric whose optimized threshold is used to make class predictions from probability predictions.
        early_stopping_metric : Optional[BinaryClassificationMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[List[BinaryClassificationMetric]]
            A list of metrics to evaluate the trained models.
        name : Optional[str]
            The name of the task.
        """
        name = name if name is not None else f"{self.__class__.__name__}('target_column'={repr(target_column)})"

        super().__init__(
            hps_tuning_metric=hps_tuning_metric,
            name=name,
            target_column=target_column,
            criterion=criterion,
            early_stopping_metric=early_stopping_metric,
            evaluation_metrics=evaluation_metrics
        )

        self._validate_metrics_type(type_=BinaryClassificationMetric)
        self._validate_criterion_type(type_=BinaryClassificationLoss)

        self._set_decision_threshold_metric(decision_threshold_metric)

    @property
    def criterion(self) -> Optional[BinaryClassificationLoss]:
        return self._criterion

    @property
    def decision_threshold_metric(self) -> BinaryClassificationMetric:
        return self._decision_threshold_metric

    @property
    def early_stopping_metric(self) -> BinaryClassificationMetric:
        return self._early_stopping_metric  # type: ignore

    @property
    def evaluation_metrics(self) -> List[BinaryClassificationMetric]:
        return self._evaluation_metrics  # type: ignore

    @property
    def hps_tuning_metric(self) -> BinaryClassificationMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> List[BinaryClassificationMetric]:
        return self._metrics  # type: ignore

    def _set_decision_threshold_metric(self, decision_threshold_metric: BinaryClassificationMetric):
        if decision_threshold_metric:
            self._decision_threshold_metric = decision_threshold_metric
            if decision_threshold_metric.name not in self._unique_metric_names:
                self._unique_metric_names.add(decision_threshold_metric.name)
                self._metrics.append(decision_threshold_metric)
        else:
            self._decision_threshold_metric = self._hps_tuning_metric


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
            evaluation_metrics: Optional[List[RegressionMetric]] = None,
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
        evaluation_metrics : Optional[List[RegressionMetric]]
            A list of metrics to evaluate the trained models.
        name : Optional[str]
            The name of the task.
        """
        name = name if name is not None else f"{self.__class__.__name__}('target_column'={repr(target_column)})"

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
    def early_stopping_metric(self) -> RegressionMetric:
        return self._early_stopping_metric    # type: ignore

    @property
    def evaluation_metrics(self) -> List[RegressionMetric]:
        return self._evaluation_metrics    # type: ignore

    @property
    def hps_tuning_metric(self) -> RegressionMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> List[RegressionMetric]:
        return super().metrics  # type: ignore


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
            evaluation_metrics: Optional[List[SegmentationMetric]] = None,
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
        evaluation_metrics : Optional[List[SegmentationMetric]]
            A list of metrics to evaluate the trained models.
        name : Optional[str]
            The name of the task.
        """
        default_name = f"{self.__class__.__name__}('modality'={repr(modality)}, 'organ'={repr(organ)})"
        name = name if name is not None else default_name

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
    def early_stopping_metric(self) -> SegmentationMetric:
        return self._early_stopping_metric  # type: ignore

    @property
    def evaluation_metrics(self) -> List[SegmentationMetric]:
        return self._evaluation_metrics  # type: ignore

    @property
    def hps_tuning_metric(self) -> SegmentationMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> List[SegmentationMetric]:
        return super().metrics  # type: ignore
