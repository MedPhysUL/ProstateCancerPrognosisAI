"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the multi-task losses used to measure models' performance. The
                        ultimate goal is to implement a loss based on uncertainty. See :
                            1. https://towardsdatascience.com/deep-multi-task-learning-3-lessons-learned-7d0193d71fd6
                            2. https://arxiv.org/pdf/1705.07115.pdf
                            3. https://github.com/yaringal/multi-task-learning-example
"""

from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Dict, List, Optional, Union

from torch import Tensor

from ...tasks.base import Task
from ...tasks.containers import TaskList


class MultiTaskLoss(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as multi-task optimization criteria.
    """

    TORCH_LIKE_SERIALIZABLE_ATTRIBUTES = ["tasks"]

    def __init__(
            self,
            name: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : Optional[str]
            Name of the Loss.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks in the dataset.
        """
        self.name = name if name is not None else f"{self.__class__.__name__}"
        self.tasks = TaskList(tasks)

    def __call__(
            self,
            losses: Dict[str, Tensor]
    ) -> Tensor:
        """
        Gets multi-task loss value.

        Parameters
        ----------
        losses : Dict[str, Tensor]
            Dictionary of single task losses. Keys are task names and values are losses (1, 1) tensors.

        Returns
        -------
        loss : Tensor
            (1, 1) tensor.
        """
        assert self.tasks is not None, "The 'tasks' attribute must be set before computing the multi-task loss."

        return self._compute_loss(losses)

    @abstractmethod
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
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state of the multi-task loss.

        Returns
        -------
        states: Dict[str, Any]
            The state of the task.
        """
        state = {}

        for k, v in vars(self).items():
            if k in self.TORCH_LIKE_SERIALIZABLE_ATTRIBUTES:
                if v:
                    state[k] = v.state_dict()
                else:
                    state[k] = None
            else:
                state[k] = copy(v)

        return state
