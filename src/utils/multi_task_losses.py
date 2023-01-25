"""
    @file:              multi_task_losses.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 01/2023

    @Description:       This file is used to define the multi-task losses used to measure models' performance. The
                        ultimate goal is to implement a loss based on uncertainty. See :
                            1. https://towardsdatascience.com/deep-multi-task-learning-3-lessons-learned-7d0193d71fd6
                            2. https://arxiv.org/pdf/1705.07115.pdf
                            3. https://github.com/yaringal/multi-task-learning-example
"""


from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from torch import nanmean, stack, Tensor

from src.utils.tasks import Task


class MultiTaskLoss(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as multi-task optimization criteria.
    """

    def __init__(
            self,
            name: str,
            tasks: Optional[List[Task]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : str
            Name of the Loss.
        tasks : Optional[List[Task]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks.
        """
        # Protected attributes
        self._name = name if name is not None else f"{self.__class__.__name__}"
        self._tasks = tasks

    @property
    def name(self) -> str:
        return self._name

    @property
    def tasks(self) -> Optional[List[Task]]:
        return self._tasks

    @tasks.setter
    def tasks(self, tasks: List[Task]) -> None:
        self._tasks = tasks

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


class MeanLoss(MultiTaskLoss):
    """
    Callable class that computes the mean loss between all tasks. This is the most simple possible multi-task loss.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            tasks: Optional[List[Task]] = None
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : Optional[str]
            Name of the multi-task loss.
        tasks : Optional[List[Task]]
            Tasks to include in the multi-task loss calculation. By default, we use all available tasks.
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
