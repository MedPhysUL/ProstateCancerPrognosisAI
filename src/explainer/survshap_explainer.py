"""
    @file:              survshap_explainer.py
    @Author:            Felix Desroches

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file contains a class used to analyse and explain how a model works using shapley values
    for survival tasks.
"""

import os
from typing import Dict, List, Optional, Union, Tuple, NamedTuple

import pandas
import shap
import sklearn.utils._mocking
import survshap
from survshap.model_explanations.plot import model_plot_mean_abs_shap_values, model_plot_shap_lines_for_all_individuals, model_plot_shap_lines_for_variables
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
import torch

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..explainer.shap_explainer import TableShapValueExplainer
from ..models.base.model import Model
from ..tasks.base import Task, TableTask
from ..tasks.containers import TaskList
from ..tasks.survival_analysis import SurvivalAnalysisTask
from ..tools.transforms import to_numpy


class SurvshapTuple(NamedTuple):
    x: float
    y: float


class SurvshapWrapper:
    def __init__(
            self,
            function
    ):
        self.function = function

    def __call__(
            self,
            *args,
            **kwargs
    ):
        return self.function


class TableSurvshapExplainer:
    """
    This class allows the user to interpret a model using a time-dependant analysis of survival tasks with shap values.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,

    ):
        self.model = model
        self.dataset = dataset
        self.predictions_dict = {k: to_numpy(v) for k, v in self.model.predict_on_dataset(dataset=self.dataset).items()}

    @staticmethod
    def _get_structured_array(
            event_indicator: np.ndarray,
            event_time: np.ndarray
    ) -> np.ndarray:
        """
        Returns a structured array with event indicator and event time.

        Parameters
        ----------
        event_indicator : np.ndarray
            (N,) array with event indicator.
        event_time : np.ndarray
            (N,) array with event time.

        Returns
        -------
        structured_array : np.ndarray
            (N, 2) structured array with event indicator and event time.
        """
        structured_array = np.empty(shape=(len(event_indicator),), dtype=[('event', int), ('time', float)])
        structured_array['event'] = event_indicator.astype(bool)
        structured_array['time'] = event_time

        return structured_array

    def compute_explanation(
            self,
            task: SurvivalAnalysisTask,
            function: str
    ):
        prediction = {}
        predictions = PredictionEvaluator.slice_patient_dictionary(self.predictions_dict, separate_patients=True)
        for prediction_element in predictions:
            if prediction.get(task.name, None) is not None:
                prediction[task.name] = np.concatenate((prediction.get(task.name), prediction_element[task.name]))
            else:
                prediction[task.name] = (prediction_element[task.name])
        assert function == "chf" or function == "sf", "Only the survival function ('sf') and cumulative hazard " \
                                                      "function ('chz') are implemented."
        if function == "chf":
            wrapper = SurvshapWrapper(task.breslow_estimator.get_cumulative_hazard_function(prediction[task.name]))
        elif function == "sf":
            wrapper = SurvshapWrapper(task.breslow_estimator.get_survival_function(prediction[task.name]))
        explainer = survshap.SurvivalModelExplainer(self.model,
                                                    pandas.DataFrame(
                                                        to_numpy(self.dataset.table_dataset.x),
                                                        columns=self.dataset.table_dataset.features_columns
                                                    ),
                                                    self._get_structured_array(to_numpy(
                                                        self.dataset.table_dataset.y[task.name][:, 0]),
                                                        to_numpy(self.dataset.table_dataset.y[task.name][:, 1])),
                                                    predict_cumulative_hazard_function=wrapper)
        survshap_explanation = survshap.ModelSurvSHAP(function_type=function, random_state=11121)

        survshap_explanation.fit(explainer)

        return survshap_explanation

    def plot_shap_lines_for_all_patients(
            self,
            feature: str,
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None
    ):
        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )

        for task in tasks:
            explainer = self.compute_explanation(task=task, function=function)
            model_plot_shap_lines_for_all_individuals(
                full_result=explainer.full_result,
                timestamps=explainer.timestamps,
                event_inds=explainer.event_ind,
                event_times=explainer.event_times,
                variable=feature
            )

    def plot_shap_lines_for_features(
            self,
            features: List[str],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]],
            discretise: Optional[Tuple[int, List[str]]] = None
    ):
        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )

        if discretise is not None:
            method = "quantile"
        else:
            method = ""
            discretise = (5, [])

        for task in tasks:
            explainer = self.compute_explanation(task=task, function=function)
            model_plot_shap_lines_for_variables(
                full_result=explainer.full_result,
                timestamps=explainer.timestamps,
                event_inds=explainer.event_ind,
                event_times=explainer.event_times,
                variables=features,
                to_discretize=discretise[1],
                discretization_method=method,
                n_bins=discretise[0],
                show=True
            )

    def plot_shap_average_of_absolute_value(
            self,
            features: List[str],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]]
    ):
        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )

        for task in tasks:
            explainer = self.compute_explanation(task=task, function=function)
            model_plot_mean_abs_shap_values(
                result=explainer.result,
                timestamps=explainer.timestamps,
                event_inds=explainer.event_ind,
                event_times=explainer.event_times,
                variables=features
            )
