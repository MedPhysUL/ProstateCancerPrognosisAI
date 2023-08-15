"""
    @file:              prediction_evaluator.py
    @Author:            Félix Desroches

    @Creation Date:     08/2023
    @Last modification: 08/2023

    @Description:       This file contains a class used to compare two models and compute the p-values for different
    metrics.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import scipy
import torch

from ..data.datasets.prostate_cancer import ProstateCancerDataset
from ..models.base.model import Model
from ..tasks.base import Task
from ..tasks.containers.list import TaskList, SurvivalAnalysisTask, BinaryClassificationTask, TableTask
from ..tools.transforms import to_numpy

from ..tools.delong_test import delong_roc_test
from compare_concordance import compare_concordance


class ModelComparator:
    """
    This class is used to compute the different p-values for metrics used in the comparison of two different models.
    """

    def __init__(
            self,
            model_1: Model,
            model_2: Model,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Sets the required attributes.

        Parameters
        ----------
        model_1 : Model
            The first model to compare.
        model_2 : Model
            The second model to compare.
        dataset : ProstateCancerDataset
            The dataset on which to compare the models.
        """
        self.model_1 = model_1
        self.model_2 = model_2
        self.dataset = dataset

    def compute_c_index_p_value(
            self,
            mask: Optional[List[int]] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Computes the p-value for the concordance index censored.

        Parameters
        ----------
        mask : Optional[List[int]]
            The mask used to select the patients with which to compute the p-value. Defaults to all patients.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            A list of tasks for which to compute the p-value. Defaults to all survival analysis tasks.

        Returns
        -------
        p_value : Dict[str, Union[float, int]]
            The computed p-value for each task.
        """
        results_1 = self.model_1.predict_on_dataset(dataset=self.dataset, mask=mask)
        results_2 = self.model_2.predict_on_dataset(dataset=self.dataset, mask=mask)

        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, SurvivalAnalysisTask) for task in tasks), (
                f"All tasks must be instances of 'SurvivalAnalysisTask'."
            )
        p_values = {}
        for task in tasks:
            p_values[task.name] = compare_concordance(
                self.dataset.table_dataset.y[task.name][:, 1].tolist(),
                self.dataset.table_dataset.y[task.name][:, 0].tolist(),
                results_1[task.name],
                results_2[task.name]
            )[4]

        return p_values

    def compute_auc_p_value(
            self,
            mask: Optional[List[int]] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ):
        """

        """
        results_1 = self.model_1.predict_on_dataset(dataset=self.dataset, mask=mask)
        results_2 = self.model_2.predict_on_dataset(dataset=self.dataset, mask=mask)

        if tasks is None:
            tasks = self.dataset.tasks.binary_classification_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, BinaryClassificationTask) for task in tasks), (
                f"All tasks must be instances of 'BinaryClassificationTask'."
            )
        p_values = {}
        for task in tasks:
            p_values[task.name] = 10**delong_roc_test(
                np.expand_dims(self.dataset.table_dataset.y[task.name], 1),
                to_numpy(torch.squeeze(results_1[task.name])),
                to_numpy(torch.squeeze(results_2[task.name]))
            )[0, 0]
        return p_values

    def compute_variance_p_value(
            self,
            mask: Optional[List[int]] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ):
        """

        """
        results_1 = self.model_1.predict_on_dataset(dataset=self.dataset, mask=mask)
        results_2 = self.model_2.predict_on_dataset(dataset=self.dataset, mask=mask)

        if tasks is None:
            tasks = self.dataset.tasks.binary_classification_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )
        p_values = {}
        for task in tasks:
            p_values[task.name] = scipy.stats.levene(
                to_numpy(torch.squeeze(results_1[task.name])),
                to_numpy(torch.squeeze(results_2[task.name]))
            ).pvalue
        return p_values

    def compute_mean_p_value(
            self,
            mask: Optional[List[int]] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ):
        """

        """
        results_1 = self.model_1.predict_on_dataset(dataset=self.dataset, mask=mask)
        results_2 = self.model_2.predict_on_dataset(dataset=self.dataset, mask=mask)

        if tasks is None:
            tasks = self.dataset.tasks.binary_classification_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )
        p_values = {}
        for task in tasks:
            p_values[task.name] = scipy.stats.ttest_ind(
                to_numpy(torch.squeeze(results_1[task.name])),
                to_numpy(torch.squeeze(results_2[task.name]))
            ).pvalue
        return p_values
