"""
    @file:              survshap_explainer.py
    @Author:            Felix Desroches

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file contains a class used to analyse and explain how a model works using shapley values
    for survival tasks.
"""

import os
from typing import List, Optional, Union, Tuple, NamedTuple
import pandas
import survshap
from survshap.model_explanations.plot import model_plot_mean_abs_shap_values, model_plot_shap_lines_for_all_individuals, model_plot_shap_lines_for_variables
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..models.base.model import Model
from ..tasks.base import Task, TableTask
from ..tasks.containers import TaskList
from ..tasks.survival_analysis import SurvivalAnalysisTask
from ..tools.transforms import to_numpy


class SurvshapTuple(NamedTuple):
    """
    Tuple to use with the SurvSHAP library.
    """
    x: float
    y: float


class SurvshapWrapper:
    """
    Wrapper to create a way to compute the cumulative hazard functions and survival functions using pandas DataFrames.
    """
    def __init__(
            self,
            function,
            dataset: ProstateCancerDataset,
            task: SurvivalAnalysisTask,
    ) -> None:
        """
        Sets up the required attributes.

        Parameters
        ----------
        function
            The function with which to compute the cumulative hazard or survival functions.
        dataset : ProstateCancerDataset
            The dataset with which to compute the SurvSHAP values.
        task : SurvivalAnalysisTask
            The task for which to compute the SurvSHAP values.
        """
        self.function = function
        self.dataset = dataset
        self.task = task
        self.table_order = self.dataset.table_dataset.features_columns

    def _convert_data_frame_to_features_type(
            self,
            data_frame: pandas.DataFrame,
            individual_patients: bool = False
    ) -> Union[List[FeaturesType], FeaturesType]:
        """
        Converts a pandas DataFrame to a FeaturesType object.

        Parameters
        ----------
        data_frame : pandas.DataFrame
            The dataframe to convert.
        individual_patients : bool
            Whether to separate the converted DataFrame into a list of individual FeaturesType for individual patients,
            defaults to false.

        Returns
        -------
        features_type  : Union[List[FeaturesType], FeaturesType]
            The converted DataFrame.
        """
        if not individual_patients:
            table_data = {}
            if data_frame.ndim == 2:
                for feature in self.table_order:
                    table_data[feature] = torch.unsqueeze(torch.tensor(data_frame.loc[:, feature]), -1)
            elif data_frame.ndim == 1:
                for feature in self.table_order:
                    table_data[feature] = torch.unsqueeze(torch.tensor(data_frame.loc[feature]), -1)
            return FeaturesType({}, table_data)
        else:
            patients_list = []
            for i in range(data_frame.shape[0]):
                patients_list.append(self._convert_data_frame_to_features_type(data_frame.iloc[i, :]))
            return patients_list

    def _convert_features_type_to_data_frame(
            self,
            features_type: FeaturesType
    ) -> pandas.DataFrame:
        """
        Converts a FeaturesType object to a pandas DataFrame.

        Parameters
        ----------
        features_type : FeaturesType
            The FeaturesType to convert.

        Returns
        -------
        dataframe : pandas.DataFrame
            The converted FeaturesType
        """
        i = 1
        concatenated = torch.tensor([])
        for feature in self.table_order:
            datum = features_type.table[feature]
            if isinstance(datum, torch.Tensor):
                datum = to_numpy(datum)
            elif isinstance(datum, np.ndarray):
                pass
            else:
                datum = np.array([datum])
            datum = np.expand_dims(datum, 1)
            if i == 1:
                concatenated = datum
            else:
                concatenated = np.concatenate((concatenated, datum), 1)
            i += 1
        return pandas.DataFrame(concatenated, columns=self.table_order)

    def __call__(
            self,
            model: Model,
            data: pandas.DataFrame
    ) -> np.ndarray:
        """
        Computes the desired stepfunctions.

        Parameters
        ----------
        model : Model
            The model with which to compute the predictions and for which to determine the SurvSHAP values.
        data : pandas.DataFrame
            The modified data with which to commpute the predictions in the form of a DataFrame.

        Returns
        -------
        function : np.ndarray
            The computed cumulative hazard or survival function.
        """
        features = self._convert_data_frame_to_features_type(data, True)
        prediction = {}
        predictions = [model.predict(feature) for feature in features]
        for prediction_element in predictions:
            if prediction.get(self.task.name, None) is not None:
                prediction[self.task.name] = np.concatenate(
                    (prediction.get(self.task.name).cpu(), prediction_element[self.task.name].cpu())
                )
            else:
                prediction[self.task.name] = (prediction_element[self.task.name])
        return self.function(prediction[self.task.name])


class TableSurvshapExplainer:
    """
    This class allows the user to interpret a model using a time-dependant analysis of survival tasks with shap values.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Sets up the required attributes.

        Parameters
        ----------
        model : Model
            The model to explain.
        dataset : ProstateCancerDataset
            The dataset to use to compute the SurvSHAP values
        """
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
    ) -> survshap.ModelSurvSHAP:
        """
        Computes the SurvSHAP explanations to use with the SurvSHAP graphs.

        Parameters
        ----------
        task : SurvivalAnalysisTask
            The task for which to compute the explanation.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.

        Returns
        -------
        survshap_explanation : survshap.ModelSurvSHAP
            The computed explanation.
        """
        assert function == "chf" or function == "sf", "Only the survival function ('sf') and cumulative hazard " \
                                                      "function ('chz') are implemented."
        if function == "chf":
            wrapper = SurvshapWrapper(task.breslow_estimator.get_cumulative_hazard_function, self.dataset, task)
        elif function == "sf":
            wrapper = SurvshapWrapper(task.breslow_estimator.get_survival_function, self.dataset, task)
        else:
            wrapper = None
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
    ) -> None:
        """
        Plots the Shap values of a single feature as a function of time for all patients.

        Parameters
        ----------
        feature : str
            The name of the feature for which to compute the graph.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        """
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
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            discretise: Optional[Tuple[int, List[str]]] = None
    ) -> None:
        """
        Plots the Shap values of each desired feature as a function of time for all patients.

        Parameters
        ----------
        features : List[str]
            The names of the features for which to compute the graph.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        discretise : Optional[Tuple[int, List[str]]]
            If descretising a feature is desired, then it is the Tuple of the number of bins and a list of the features
            to discretise.
        """
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
    ) -> None:
        """
        Plots the average of the absolute value of each Shap value as a function of time.

        Parameters
        ----------
        features : List[str]
            The names of the features for which to compute the graph.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        """
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
