"""
    @file:              list.py
    @Author:            Maxence Larose

    @Creation Date:     02/2022
    @Last modification: 02/2022

    @Description:       This file is used to define the `TaskList` class which essentially acts as a list of
                        tasks.
"""

from __future__ import annotations
from typing import Any, Dict, Generic, Iterable, Iterator, Optional, TypeVar, Union

from ..base import TableTask, Task
from ...tasks import BinaryClassificationTask, RegressionTask, SegmentationTask, SurvivalAnalysisTask

_SpecifiedTaskType = TypeVar("_SpecifiedTaskType")


class TaskList(Generic[_SpecifiedTaskType]):
    """
    Holds tasks in a list.
    """

    def __init__(self, tasks: Optional[Union[Task, Iterable[Task]]] = None):
        """
        Constructor of the TaskList class.

        Parameters
        ----------
        tasks : Optional[Iterable[Task]]
            The tasks to use.
        """
        if tasks is None:
            tasks = []
        if isinstance(tasks, Task):
            tasks = [tasks]

        assert isinstance(tasks, Iterable), "tasks must be an Iterable."
        assert all(isinstance(task, Task) for task in tasks), "All tasks must be instances of Task."

        self.tasks = list(tasks)
        self.check_for_duplicate_task_names()

    def __getitem__(self, idx: int) -> Union[_SpecifiedTaskType, Task]:
        """
        Gets a task from the list.

        Parameters
        ----------
        idx : int
            The index of the task to get.

        Returns
        -------
        task : Union[_SpecifiedTaskType, Task]
            The task at the given index in the list of tasks.
        """
        return self.tasks[idx]

    def __iter__(self) -> Iterator[Union[_SpecifiedTaskType, Task]]:
        """
        Gets an iterator over the tasks.

        Returns
        -------
        iterator : Iterator[Task]
            An iterator over the tasks.
        """
        return iter(self.tasks)

    def __len__(self) -> int:
        """
        Gets the number of tasks in the list.

        Returns
        -------
        number : int
            The number of tasks in the list.
        """
        return len(self.tasks)

    def __add__(self, other: TaskList) -> TaskList:
        """
        Adds another TaskList to the current TaskList.

        Parameters
        ----------
        other : TaskList
            Another TaskList.

        Returns
        -------
        task_list : TaskList
            Augmented TaskList.
        """
        return TaskList(self.tasks + other.tasks)

    @property
    def binary_classification_tasks(self) -> TaskList[BinaryClassificationTask]:
        """
        Returns a 'TaskList' of all binary classification tasks contained in the current 'TaskList'.

        Returns
        -------
        binary_classification_tasks : TaskList[BinaryClassificationTask]
            Binary classification tasks.
        """
        return TaskList([task for task in self.tasks if isinstance(task, BinaryClassificationTask)])

    @property
    def regression_tasks(self) -> TaskList[RegressionTask]:
        """
        Returns a 'TaskList' of all regression tasks contained in the current 'TaskList'.

        Returns
        -------
        regression_tasks : TaskList[RegressionTask]
            Regression tasks.
        """
        return TaskList([task for task in self.tasks if isinstance(task, RegressionTask)])

    @property
    def segmentation_tasks(self) -> TaskList[SegmentationTask]:
        """
        Returns a 'TaskList' of all segmentation tasks contained in the current 'TaskList'.

        Returns
        -------
        segmentation_tasks : TaskList[SegmentationTask]
            Segmentation tasks.
        """
        return TaskList([task for task in self.tasks if isinstance(task, SegmentationTask)])

    @property
    def survival_analysis_tasks(self) -> TaskList[SurvivalAnalysisTask]:
        """
        Returns a 'TaskList' of all survival analysis tasks contained in the current 'TaskList'.

        Returns
        -------
        survival_tasks : TaskList[SurvivalAnalysisTask]
            Survival analysis tasks.
        """
        return TaskList([task for task in self.tasks if isinstance(task, SurvivalAnalysisTask)])

    @property
    def table_tasks(self) -> TaskList[TableTask]:
        """
        Returns a 'TaskList' of all table tasks contained in the current 'TaskList'.

        Returns
        -------
        table_tasks : TaskList[RegressionTask]
            Table tasks.
        """
        return TaskList([task for task in self.tasks if isinstance(task, TableTask)])

    def append(self, task: Task):
        """
        Appends a task to the list.

        Parameters
        ----------
        task : Task
            The task to append.
        """
        assert isinstance(task, Task), "task must be an instance of 'Task'."
        self.tasks.append(task)
        self.check_for_duplicate_task_names()

    def check_for_duplicate_task_names(self):
        """
        Checks if there is any duplicates in the 'TaskList'.
        """
        seen, duplicates = [], []

        for task in self.tasks:
            if task.name in seen:
                duplicates.append(task.name)
            else:
                seen.append(task.name)

        if duplicates:
            raise AssertionError(f"Duplicates 'Task' are not allowed in 'TaskList'. Duplicates are defined based on "
                                 f"task name. Found duplicates {duplicates}.")

    def remove(self, task: Task):
        """
        Removes a task from the list.

        Parameters
        ----------
        task : Task
            The task to remove.
        """
        assert isinstance(task, Task), "task must be an instance of 'Task'."
        self.tasks.remove(task)

    def state_dict(self) -> Dict[str, Any]:
        """
        Collates the states of the tasks in a dictionary.

        Returns
        -------
        states: Dict[str, Any]
            The state of the tasks.
        """
        return {task.name: task.state_dict() for task in self.tasks}
