"""
    @file:              tasks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the different possible tasks.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

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
            criterion: Optional[Loss] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        optimization_metric : Metric
            A score metric. This metric is used for Optuna hyperparameters optimization.
        criterion : Optional[Callable]
            A loss function.
        """
        self._optimization_metric = optimization_metric
        self._criterion = criterion

    @property
    def criterion(self) -> Optional[Loss]:
        return self._criterion

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
            criterion: Optional[Union[BinaryClassificationLoss, RegressionLoss]] = None
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
        """
        self._target_col = target_col

        super().__init__(criterion=criterion, optimization_metric=optimization_metric)

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
            criterion: Optional[BinaryClassificationLoss] = None
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
        """
        super().__init__(optimization_metric=optimization_metric, target_col=target_col, criterion=criterion)

    @property
    def name(self) -> str:
        return f"{self._target_col}_classification"

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
            criterion: Optional[RegressionLoss] = None
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
        """
        super().__init__(optimization_metric=optimization_metric, target_col=target_col, criterion=criterion)

    @property
    def name(self) -> str:
        return f"{self._target_col}_regression"

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
            modality: str
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
        """
        self._organ = organ
        self._modality = modality
        super().__init__(criterion=criterion, optimization_metric=optimization_metric)

    @property
    def organ(self) -> str:
        return self._organ

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def name(self) -> str:
        return f"{self._organ}_segmentation"

    @property
    def task_type(self) -> TaskType:
        return TaskType.SEGMENTATION
