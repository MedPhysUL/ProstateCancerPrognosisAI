"""
    @file:              prediction_comparator.py
    @Author:            Maxence Larose, Félix Desroches

    @Creation Date:     09/2023
    @Last modification: 09/2023

    @Description:       This file contains a class used to compare two models and compute the p-values for different
                        metrics. Any problems with compareC? Use the following line:
                            gcc -shared -o compareC.so -fPIC compareC.c
                        to compile the compareC.c file.
"""

from typing import Dict, List, Optional, Union
import warnings

from compare_concordance import compare_concordance
import numpy as np
import scipy

from ..metrics.single_task.base import SingleTaskMetric
from ..metrics.single_task.binary_classification import BinaryClassificationMetric
from ..metrics.single_task.survival_analysis import SurvivalAnalysisMetric
from ..data.datasets.prostate_cancer import TargetsType
from ..tasks.base import Task
from ..tasks.containers.list import TaskList
from ..tools.delong_test import delong_roc_test


class PredictionComparator:
    """
    This class is used to compute the different p-values for metrics used in the comparison of two different models.
    """

    def __init__(
            self,
            pred_1: TargetsType,
            pred_2: TargetsType,
            ground_truth: TargetsType,
            tasks: Optional[Union[Task, TaskList, List[Task]]]
    ) -> None:
        """
        Sets the required attributes.

        Parameters
        ----------
        pred_1 : TargetsType
            The first pred to compare.
        pred_2 : TargetsType
            The second pred to compare.
        ground_truth : TargetsType
            The ground truth.
        """
        self._pred_1 = pred_1
        self._pred_2 = pred_2
        self._ground_truth = ground_truth

        task_names = list(pred_1.keys())
        tasks = TaskList(tasks)
        binary_classification_task_names = [t.name for t in tasks.binary_classification_tasks]
        survival_analysis_task_names = [t.name for t in tasks.survival_analysis_tasks]

        self._binary_classification_task_names = [t for t in task_names if t in binary_classification_task_names]
        self._survival_analysis_task_names = [t for t in task_names if t in survival_analysis_task_names]

    def compute_c_index_p_value(self,) -> Dict[str, Union[int, float]]:
        p_values = {}
        for task_name in self._survival_analysis_task_names:
            p_values[task_name] = compare_concordance(
                self._ground_truth[task_name][:, 1].tolist(),
                self._ground_truth[task_name][:, 0].tolist(),
                self._pred_1[task_name],
                self._pred_2[task_name]
            )[4]

        return p_values

    def compute_auc_p_value(self):
        p_values = {}
        for task_name in self._binary_classification_task_names:
            p_values[task_name] = 10**delong_roc_test(
                np.array(self._ground_truth[task_name]),
                np.array(self._pred_1[task_name]),
                np.array(self._pred_2[task_name])
            )[0, 0]
        return p_values

    def compute_variance_p_value(self):
        p_values = {}
        for task_name in self._binary_classification_task_names + self._survival_analysis_task_names:
            p_values[task_name] = scipy.stats.levene(
                self._pred_1[task_name],
                self._pred_2[task_name]
            ).pvalue
        return p_values

    def compute_mean_p_value(self):
        p_values = {}
        for task_name in self._binary_classification_task_names + self._survival_analysis_task_names:
            p_values[task_name] = scipy.stats.ttest_ind(
                self._pred_1[task_name],
                self._pred_2[task_name]
            ).pvalue
        return p_values

    def compute_any_metric_p_value(
            self,
            metric: Union[SingleTaskMetric, Dict[str, SingleTaskMetric]],
            n_samples: int = 10_000
    ):
        warnings.simplefilter('ignore')
        if isinstance(metric, dict):
            tasks = list(metric.keys())
        else:
            if isinstance(metric, BinaryClassificationMetric):
                tasks = self._binary_classification_task_names
            elif isinstance(metric, SurvivalAnalysisMetric):
                tasks = self._survival_analysis_task_names
            else:
                raise ValueError(f"Metric {metric} not supported.")

        p_values = {}
        for task_name in tasks:
            _metric = metric[task_name] if isinstance(metric, dict) else metric
            pred_1, pred_2 = np.array(self._pred_1[task_name]), np.array(self._pred_2[task_name])
            ground_truth = np.array(self._ground_truth[task_name])
            diff = _metric(pred_2, ground_truth) - _metric(pred_1, ground_truth)
            samples = []
            for _ in range(n_samples):
                try:
                    idx = np.random.choice(len(pred_1), size=len(pred_1), replace=True)
                    metric_1_bootstrap = _metric(pred_1[idx], ground_truth[idx])
                    metric_2_bootstrap = _metric(pred_2[idx], ground_truth[idx])
                    samples.append(metric_2_bootstrap - metric_1_bootstrap)
                except ValueError:
                    pass

            std = np.sqrt(np.nanvar(samples))
            z_score = diff / std
            p_value = 2 * (1.0 - scipy.stats.norm.cdf(np.abs(z_score)))

            p_values[task_name] = p_value

        return p_values
