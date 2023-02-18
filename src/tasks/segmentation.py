"""
    @file:              segmentation.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `SegmentationTask` class.
"""

from typing import Iterable, Optional, Union

from .base import Task
from ..losses.single_task.segmentation import SegmentationLoss
from ..metrics.single_task.containers import SingleTaskMetricList
from ..metrics.single_task.segmentation import SegmentationMetric


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
            evaluation_metrics: Optional[
                Union[
                    SegmentationMetric,
                    Iterable[SegmentationMetric],
                    SingleTaskMetricList[SegmentationMetric]
                ]
            ] = None,
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
        evaluation_metrics : Optional[
                Union[
                    SegmentationMetric,
                    Iterable[SegmentationMetric],
                    SingleTaskMetricList[SegmentationMetric]
                ]
            ]
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
    def evaluation_metrics(self) -> Optional[SingleTaskMetricList[SegmentationMetric]]:
        return self._evaluation_metrics

    @property
    def hps_tuning_metric(self) -> SegmentationMetric:
        return self._hps_tuning_metric  # type: ignore

    @property
    def metrics(self) -> SingleTaskMetricList[SegmentationMetric]:
        return super().metrics

    @property
    def unique_metrics(self) -> SingleTaskMetricList[SegmentationMetric]:
        return super().unique_metrics

