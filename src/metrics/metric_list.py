"""
    @file:              metric_list.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `MetricList` class which essentially acts as a list of
                        metrics.
"""

from __future__ import annotations
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, Union

from src.metrics.metric import Metric

_SpecifiedMetricType = TypeVar("_SpecifiedMetricType")


class MetricList(Generic[_SpecifiedMetricType]):
    """
    Holds metrics in a list.
    """

    def __init__(
            self,
            metrics: Optional[Union[Metric, Iterable[Metric]]] = None
    ):
        """
        Constructor of the MetricList class.

        Parameters
        ----------
        metrics : Optional[Union[Metric, Iterable[Metric]]]
            The metrics to use.
        """
        if metrics is None:
            metrics = []
        if isinstance(metrics, Metric):
            metrics = [metrics]

        assert isinstance(metrics, Iterable), "metrics must be an Iterable."
        assert all(isinstance(metric, Metric) for metric in metrics), "All metrics must be instances of Metric."

        self.metrics = list(metrics)

    def __getitem__(self, idx: int) -> Union[Metric, _SpecifiedMetricType]:
        """
        Gets a metric from the list.

        Parameters
        ----------
        idx : int
            The index of the metric to get.

        Returns
        -------
        metric : Union[Metric, _SpecifiedMetricType]
            The metric at the given index in the list of metrics.
        """
        return self.metrics[idx]

    def __iter__(self) -> Iterator[Union[Metric, _SpecifiedMetricType]]:
        """
        Gets an iterator over the metrics.

        Returns
        -------
        iterator : Iterator[Union[Metric, _SpecifiedMetricType]]
            An iterator over the metrics.
        """
        return iter(self.metrics)

    def __len__(self) -> int:
        """
        Gets the number of metrics in the list.

        Returns
        -------
        number : int
            The number of metrics in the list.
        """
        return len(self.metrics)

    def __add__(self, other: MetricList) -> MetricList:
        """
        Adds another MetricList to the current MetricList.

        Parameters
        ----------
        other : MetricList
            Another MetricList.

        Returns
        -------
        metric_list : MetricList
            Augmented MetricList.
        """
        return MetricList(self.metrics + other.metrics)

    def append(self, metric: Metric):
        """
        Append a metric to the list.

        Parameters
        ----------
        metric : Metric
            The metric to append.
        """
        assert isinstance(metric, Metric), "metric must be an instance of 'Metric'."
        self.metrics.append(metric)

    def remove(self, metric: Metric):
        """
        Removes a metric from the list.

        Parameters
        ----------
        metric : Metric
            The metric to remove.
        """
        assert isinstance(metric, Metric), "metric must be an instance of 'Metric'."
        self.metrics.remove(metric)

    def state_dict(self) -> Dict[str, Any]:
        """
        Collates the states of the metrics in a dictionary.

        Returns
        -------
        states: Dict[str, Any]
            The state of the metrics.
        """
        return {metric.name: vars(metric).copy() for metric in self.metrics}
