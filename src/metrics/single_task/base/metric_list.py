"""
    @file:              metric_list.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the `SingleTaskMetricList` class which essentially acts as a list of
                        metrics.
"""

from __future__ import annotations
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, Union

from .metric import SingleTaskMetric

_SpecifiedMetricType = TypeVar("_SpecifiedMetricType")


class SingleTaskMetricList(Generic[_SpecifiedMetricType]):
    """
    Holds metrics in a list.
    """

    def __init__(
            self,
            metrics: Optional[Union[SingleTaskMetric, Iterable[SingleTaskMetric]]] = None
    ):
        """
        Constructor of the SingleTaskMetricList class.

        Parameters
        ----------
        metrics : Optional[Union[SingleTaskMetric, Iterable[Metric]]]
            The metrics to use.
        """
        if metrics is None:
            metrics = []
        if isinstance(metrics, SingleTaskMetric):
            metrics = [metrics]

        assert isinstance(metrics, Iterable), "metrics must be an Iterable."
        assert all(isinstance(metric, SingleTaskMetric) for metric in metrics), (
            "All metrics must be instances of SingleTaskMetric."
        )

        self.metrics = list(metrics)

    def __getitem__(self, idx: int) -> Union[SingleTaskMetric, _SpecifiedMetricType]:
        """
        Gets a metric from the list.

        Parameters
        ----------
        idx : int
            The index of the metric to get.

        Returns
        -------
        metric : Union[SingleTaskMetric, _SpecifiedMetricType]
            The metric at the given index in the list of metrics.
        """
        return self.metrics[idx]

    def __iter__(self) -> Iterator[Union[SingleTaskMetric, _SpecifiedMetricType]]:
        """
        Gets an iterator over the metrics.

        Returns
        -------
        iterator : Iterator[Union[SingleTaskMetric, _SpecifiedMetricType]]
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

    def __add__(self, other: SingleTaskMetricList) -> SingleTaskMetricList:
        """
        Adds another MetricList to the current MetricList.

        Parameters
        ----------
        other : SingleTaskMetricList
            Another SingleTaskMetricList.

        Returns
        -------
        metric_list : SingleTaskMetricList
            Augmented SingleTaskMetricList.
        """
        return SingleTaskMetricList(self.metrics + other.metrics)

    def append(self, metric: SingleTaskMetric):
        """
        Append a metric to the list.

        Parameters
        ----------
        metric : SingleTaskMetric
            The metric to append.
        """
        assert isinstance(metric, SingleTaskMetric), "metric must be an instance of 'SingleTaskMetric'."
        self.metrics.append(metric)

    def remove(self, metric: SingleTaskMetric):
        """
        Removes a metric from the list.

        Parameters
        ----------
        metric : SingleTaskMetric
            The metric to remove.
        """
        assert isinstance(metric, SingleTaskMetric), "metric must be an instance of 'SingleTaskMetric'."
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
