"""
    @file:              tasks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the different possible tasks.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Union

from src.utils.score_metrics import BinaryClassificationMetric, Metric, RegressionMetric, SegmentationMetric
from src.utils.losses import BinaryClassificationLoss, Loss, RegressionLoss, SegmentationLoss


class TaskType(Enum):
    """
    Custom enum for task types.
    """
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"

    def __iter__(self):
        return iter([self.REGRESSION, self.CLASSIFICATION, self.SEGMENTATION])


class Task(ABC):
    """
    An abstract class representing a task.
    """

    def __init__(
            self,
            optimization_metric: Metric,
            criterion: Optional[Loss] = None,
            evaluation_metrics: List[Metric] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        optimization_metric : Metric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        criterion : Optional[Callable]
            A loss function.
        evaluation_metrics : List[Metric]
            A list of metrics to evaluate the trained models.
        """
        self._optimization_metric = optimization_metric
        self._criterion = criterion
        self._evaluation_metrics = evaluation_metrics

    @property
    def criterion(self) -> Optional[Loss]:
        return self._criterion

    @property
    def evaluation_metrics(self) -> Optional[List[Metric]]:
        return self._evaluation_metrics

    @property
    def optimization_metric(self) -> Metric:
        return self._optimization_metric

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        raise NotImplementedError


class TableTask(Task, ABC):
    """
    An abstract class representing a task to do on table data.
    """

    def __init__(
            self,
            optimization_metric: Metric,
            target_col: str,
            criterion: Optional[Union[BinaryClassificationLoss, RegressionLoss]] = None,
            evaluation_metrics: List[Metric] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        optimization_metric : Metric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        target_col : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[Union[BinaryClassificationLoss, RegressionLoss]]
            A loss function.
        evaluation_metrics : List[Metric]
            A list of metrics to evaluate the trained models.
        """
        self._target_col = target_col

        super().__init__(
            criterion=criterion,
            optimization_metric=optimization_metric,
            evaluation_metrics=evaluation_metrics
        )

    @property
    def target_col(self) -> str:
        return self._target_col


class ClassificationTask(TableTask):
    """
    A class used to define a Classification task.
    """

    def __init__(
            self,
            optimization_metric: BinaryClassificationMetric,
            target_col: str,
            criterion: Optional[BinaryClassificationLoss] = None,
            evaluation_metrics: List[BinaryClassificationMetric] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        optimization_metric : BinaryClassificationMetric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        target_col : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[BinaryClassificationLoss]
            A loss function.
        evaluation_metrics : List[BinaryClassificationMetric]
            A list of metrics to evaluate the trained models.
        """
        super().__init__(
            optimization_metric=optimization_metric,
            target_col=target_col,
            criterion=criterion,
            evaluation_metrics=evaluation_metrics
        )

    @property
    def name(self) -> str:
        return self._target_col

    @property
    def task_type(self) -> TaskType:
        return TaskType.CLASSIFICATION


class RegressionTask(TableTask):
    """
    A class used to define a Regression task.
    """

    def __init__(
            self,
            optimization_metric: RegressionMetric,
            target_col: str,
            criterion: Optional[RegressionLoss] = None,
            evaluation_metrics: List[RegressionMetric] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        optimization_metric : RegressionMetric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        target_col : str
            Name of the column containing the targets associated to this task.
        criterion : Optional[RegressionLoss]
            A loss function.
        evaluation_metrics : List[RegressionMetric]
            A list of metrics to evaluate the trained models.
        """
        super().__init__(
            optimization_metric=optimization_metric,
            target_col=target_col,
            criterion=criterion,
            evaluation_metrics=evaluation_metrics
        )

    @property
    def name(self) -> str:
        return self._target_col

    @property
    def task_type(self) -> TaskType:
        return TaskType.REGRESSION


class SegmentationTask(Task):
    """
    A class used to define a Segmentation task.
    """

    def __init__(
            self,
            criterion: SegmentationLoss,
            optimization_metric: SegmentationMetric,
            organ: str,
            modality: str,
            evaluation_metrics: List[SegmentationMetric] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        criterion : SegmentationLoss
            A loss function.
        optimization_metric : SegmentationMetric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        organ : str
            Segmented organ.
        modality : str
            Modality on which segmentation was performed.
        evaluation_metrics : List[SegmentationMetric]
            A list of metrics to evaluate the trained models.
        """
        self._organ = organ
        self._modality = modality
        super().__init__(
            criterion=criterion,
            optimization_metric=optimization_metric,
            evaluation_metrics=evaluation_metrics
        )

    @property
    def organ(self) -> str:
        return self._organ

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def name(self) -> str:
        return self._organ

    @property
    def task_type(self) -> TaskType:
        return TaskType.SEGMENTATION
