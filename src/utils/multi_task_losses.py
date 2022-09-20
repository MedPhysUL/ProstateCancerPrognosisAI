"""
    @file:              multi_task_losses.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define the multi-task losses used to measure models' performance. The
                        ultimate goal is to implement a loss based on uncertainty. See :
                            1. https://towardsdatascience.com/deep-multi-task-learning-3-lessons-learned-7d0193d71fd6
                            2. https://arxiv.org/pdf/1705.07115.pdf
                            3. https://github.com/yaringal/multi-task-learning-example
"""


from abc import ABC, abstractmethod
from typing import List, Optional

from torch import mean, tensor, Tensor

from src.data.datasets.prostate_cancer_dataset import DataModel
from src.utils.tasks import Task


class MultiTaskLoss(ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as multi-task optimization criteria.
    """

    def __init__(
            self,
            name: str
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        name : str
            Name of the Loss.
        """
        # Protected attributes
        self._name = name
        self._tasks = None

    @abstractmethod
    def __call__(
            self,
            *args,
            **kwargs
    ) -> Tensor:
        """
        Gets loss value.

        Returns
        -------
        loss : Tensor
            (1, 1) tensor.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @property
    def tasks(self) -> Optional[List[Task]]:
        return self._tasks

    @tasks.setter
    def tasks(self, tasks: List[Task]) -> None:
        self._tasks = tasks


class MeanLoss(MultiTaskLoss):
    """
    Callable class that computes the mean loss between all tasks. This is the most simple possible multi-task loss.
    """

    def __init__(
            self,
    ):
        """
        Sets protected attributes.
        """
        super().__init__(name="MeanLoss")

    @abstractmethod
    def __call__(
            self,
            predictions: DataModel.y,
            targets: DataModel.y
    ) -> Tensor:
        """
        Gets loss value.

        Parameters
        ----------
        predictions : DataModel.y
            Batch data items.
        targets : DataElement.y
            Batch data items.

        Returns
        -------
        loss : Tensor
            (1, 1) tensor.
        """
        return mean(tensor([task.criterion(predictions[task.name], targets[task.name].float()) for task in self.tasks]))
