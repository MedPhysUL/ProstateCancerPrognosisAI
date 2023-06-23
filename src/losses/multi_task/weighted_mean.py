"""
    @file:              weighted_mean.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define the `WeightedMeanLoss` class.
"""

from ast import literal_eval
from typing import Dict, Iterable, List, Optional, Union

from torch import is_tensor, nanmean, ones, stack, tensor, Tensor

from .base import MultiTaskLoss
from ...tasks.base import Task
from ...tasks.containers import TaskList


class WeightedMeanLoss(MultiTaskLoss):
    """
    Callable class that computes the weighted mean loss between all tasks.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            weights: Optional[Union[str, Iterable[float]]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : Optional[str]
            Name of the multi-task loss.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks in the dataset.
        weights : Optional[Union[str, Iterable[float]]]
            Tasks weights for mean computation. Can also be given as a string containing the sequence of weights.
        """
        super().__init__(name=name, tasks=tasks)
        self.weights = literal_eval(weights) if isinstance(weights, str) else weights

    def _compute_loss(
            self,
            losses: Dict[str, Tensor]
    ) -> Tensor:
        """
        Gets loss value.

        Parameters
        ----------
        losses : Dict[str, Tensor]
            Dictionary of single task losses.

        Returns
        -------
        loss : Tensor
            (1, 1) tensor.
        """
        device = list(losses.values())[0].device
        if self.weights is None:
            self.weights = ones(len(self.tasks), device=device)
        elif not is_tensor(self.weights):
            self.weights = tensor(self.weights, device=device)

        return nanmean(self.weights*stack([losses[task.name] for task in self.tasks]))
