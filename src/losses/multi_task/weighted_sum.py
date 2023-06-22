"""
    @file:              weighted_sum.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define the `WeightedSumLoss` class.
"""

from typing import Dict, Iterable, List, Optional, Union

from torch import is_tensor, nansum, ones, stack, tensor, Tensor

from .base import MultiTaskLoss
from ...tasks.base import Task
from ...tasks.containers import TaskList


class WeightedSumLoss(MultiTaskLoss):
    """
    Callable class that computes the weighted sum loss between all tasks.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            weights: Optional[Iterable[float]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : Optional[str]
            Name of the multi-task loss.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks in the dataset.
        weights : Optional[Iterable[float]]
            Tasks weights for sum computation.
        """
        super().__init__(name=name, tasks=tasks)
        self.weights = weights

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

        return nansum(self.weights*stack([losses[task.name] for task in self.tasks]))
