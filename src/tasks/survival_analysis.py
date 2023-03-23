"""
    @file:              survival_analysis.py
    @Author:            Maxence Larose

    @Creation Date:     03/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the `SurvivalAnalysisTask` class.
"""

from typing import Iterable, Optional, Union

from .base import TableTask
from ..losses.single_task.survival_analysis import SurvivalAnalysisLoss
from ..metrics.single_task.containers import SingleTaskMetricList
from ..metrics.single_task.survival_analysis import SurvivalAnalysisMetric
from ..tools.missing_targets import get_idx_of_nonmissing_survival_analysis_targets


class SurvivalAnalysisTask(TableTask):
    """
    A class used to define a SurvivalAnalysisTask task.
    """

    def __init__(
            self,
            duration_column: str,
            event_column: str,
            criterion: Optional[SurvivalAnalysisLoss] = None,
            early_stopping_metric: Optional[SurvivalAnalysisMetric] = None,
            evaluation_metrics: Optional[
                Union[
                    SurvivalAnalysisMetric,
                    Iterable[SurvivalAnalysisMetric],
                    SingleTaskMetricList[SurvivalAnalysisMetric]
                ]
            ] = None,
            hps_tuning_metric: Optional[SurvivalAnalysisMetric] = None,
            name: Optional[str] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        duration_column : str
            Name of the column containing the time of event or time of censoring. In general, this is called the
            follow-up duration.
        event_column : str
            Name of the column containing the binary event indicators.
        criterion : Optional[SurvivalAnalysisLoss]
            A loss function.
        early_stopping_metric : Optional[SurvivalAnalysisMetric]
            A metric used for early stopping.
        evaluation_metrics : Optional[
                Union[
                    SurvivalAnalysisMetric,
                    Iterable[SurvivalAnalysisMetric],
                    SingleTaskMetricList[SurvivalAnalysisMetric]
                ]
            ]
            A list of metrics to evaluate the trained models.
        hps_tuning_metric : SurvivalAnalysisMetric
            A metric used for Optuna hyperparameters optimization.
        name : Optional[str]
            The name of the task.
        """
        name = name if name else f"{self.__class__.__name__}('target_column'={repr(target_column)})"

        super().__init__(
            hps_tuning_metric=hps_tuning_metric,
            name=name,
            target_column=event_column,
            criterion=criterion,
            early_stopping_metric=early_stopping_metric,
            evaluation_metrics=evaluation_metrics
        )

        self._duration_column = duration_column
        self._event_column = event_column

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_survival_analysis_targets

        self._validate_metrics_type(type_=SurvivalAnalysisMetric)
        self._validate_criterion_type(type_=SurvivalAnalysisLoss)

    @property
    def criterion(self) -> Optional[SurvivalAnalysisLoss]:
        return self._criterion

    @property
    def duration_column(self) -> str:
        return self._duration_column

    @property
    def early_stopping_metric(self) -> Optional[SurvivalAnalysisMetric]:
        return self._early_stopping_metric

    @property
    def evaluation_metrics(self) -> Optional[SingleTaskMetricList[SurvivalAnalysisMetric]]:
        return self._evaluation_metrics

    @property
    def event_column(self) -> str:
        return self._event_column

    @property
    def hps_tuning_metric(self) -> Optional[SurvivalAnalysisMetric]:
        return self._hps_tuning_metric

    @property
    def metrics(self) -> SingleTaskMetricList[SurvivalAnalysisMetric]:
        return super().metrics

    @property
    def unique_metrics(self) -> SingleTaskMetricList[SurvivalAnalysisMetric]:
        return super().unique_metrics
