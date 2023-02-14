"""
    @file:              mean.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `MeanLoss` class.
"""

from typing import Dict, List, Optional, Union

from torch import nanmean, stack, Tensor

from .base import MultiTaskLoss
from ...tasks.task import Task
from ...tasks.task_list import TaskList


class MeanLoss(MultiTaskLoss):
    """
    Callable class that computes the mean loss between all tasks.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : Optional[str]
            Name of the multi-task loss.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks in the dataset.
        """
        super().__init__(name=name, tasks=tasks)

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
        return nanmean(stack([losses[task.name] for task in self.tasks]))
