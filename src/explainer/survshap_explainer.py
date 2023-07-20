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
import survshap
from sksurv.functions import StepFunction
from survshap.model_explanations.plot import model_plot_mean_abs_shap_values, model_plot_shap_lines_for_all_individuals, model_plot_shap_lines_for_variables
import matplotlib as mpl
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


class StepFunctionWrapper(StepFunction):

    def __init__(self, step_func):
        super().__init__(x=step_func.x, y=step_func.y, a=step_func.a, b=step_func.b)

    def __call__(self, *args, **kwargs):
        old_return = super().__call__(*args, **kwargs)
        if isinstance(old_return, np.ndarray):
            return old_return
        elif isinstance(old_return, torch.Tensor):
            return super().__call__(*args, **kwargs).numpy()


class DataFrameWrapper:
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

    def _convert_ndarray_to_features_type(
            self,
            initial_array: np.ndarray,
            individual_patients: bool = False
    ) -> Union[FeaturesType, List[FeaturesType]]:
        """
        Converts a pandas DataFrame to a FeaturesType object.

        Parameters
        ----------
        initial_array : np.ndarray
            The dataframe to convert.
        individual_patients : bool
            Whether to separate the converted DataFrame into a list of individual FeaturesType for individual patients,
            defaults to false.

        Returns
        -------
        features_type  : Union[List[FeaturesType], FeaturesType]
            The converted array.
        """
        if not individual_patients:
            table_data = {}
            if initial_array.ndim == 2:
                for feature in self.table_order:
                    table_data[feature] = torch.unsqueeze(
                        torch.tensor(initial_array[:, self.table_order.index(feature)]),
                        -1
                    )
            elif initial_array.ndim == 1:
                for feature in self.table_order:
                    table_data[feature] = torch.unsqueeze(
                        torch.tensor(initial_array[self.table_order.index(feature)]),
                        -1
                    )
            return FeaturesType({}, table_data)
        else:
            patients_list = []
            for i in range(initial_array.shape[0]):
                patients_list.append(self._convert_ndarray_to_features_type(initial_array[i, :]))
            return patients_list

    def __call__(
            self,
            model: Model,
            data: Union[pandas.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the desired stepfunctions.

        Parameters
        ----------
        model : Model
            The model with which to compute the predictions and for which to determine the SurvSHAP values.
        data : Union[pandas.DataFrame, np.ndarray]
            The modified data with which to compute the predictions in the form of a DataFrame.

        Returns
        -------
        function : np.ndarray
            The computed cumulative hazard or survival function.
        """
        if isinstance(data, np.ndarray):
            features = self._convert_ndarray_to_features_type(data, True)
        else:
            features = self._convert_data_frame_to_features_type(data, True)
        prediction = {}
        predictions = [model.predict(feature) for feature in features]
        for prediction_element in predictions:
            if prediction.get(self.task.name, None) is not None:
                if (isinstance(prediction_element[self.task.name], torch.Tensor)
                        and prediction_element[self.task.name].get_device != -1):
                    to_cat = prediction_element[self.task.name].cpu()
                else:
                    to_cat = prediction_element[self.task.name]
                if (isinstance(prediction.get(self.task.name), torch.Tensor)
                        and prediction.get(self.task.name).get_device != -1):
                    prediction[self.task.name] = np.concatenate((prediction.get(self.task.name).cpu(), to_cat))
                else:
                    prediction[self.task.name] = np.concatenate((prediction.get(self.task.name), to_cat))
            else:
                prediction[self.task.name] = (prediction_element[self.task.name])
        if (isinstance(prediction[self.task.name], torch.Tensor)
                and prediction[self.task.name].get_device != -1):
            return np.array([StepFunctionWrapper(i) for i in self.function(prediction[self.task.name].cpu())])
        else:
            return np.array([StepFunctionWrapper(i) for i in self.function(prediction[self.task.name])])


class TableSurvshapExplainer:
    """
    This class allows the user to interpret a model using a time-dependant analysis of survival tasks with shap values.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
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
        self.survshap_explanation = None
        self.model = model
        self.dataset = dataset
        self.predictions_dict = {k: to_numpy(v) for k, v in self.model.predict_on_dataset(dataset=self.dataset).items()}
        self.feature_order = self.dataset.table_dataset.features_columns
        self.fitted = []

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
        if task.name not in self.fitted:
            if function == "chf":
                wrapper = DataFrameWrapper(task.breslow_estimator.get_cumulative_hazard_function, self.dataset, task)
                explainer = survshap.SurvivalModelExplainer(model=self.model,
                                                            data=pandas.DataFrame(
                                                                to_numpy(self.dataset.table_dataset.x),
                                                                columns=self.feature_order
                                                            ),
                                                            y=self._get_structured_array(to_numpy(
                                                                self.dataset.table_dataset.y[task.name][:, 0]),
                                                                to_numpy(
                                                                    self.dataset.table_dataset.y[task.name][:, 1]
                                                                )),
                                                            predict_cumulative_hazard_function=wrapper)
            elif function == "sf":
                wrapper = DataFrameWrapper(task.breslow_estimator.get_survival_function, self.dataset, task)
                explainer = survshap.SurvivalModelExplainer(model=self.model,
                                                            data=pandas.DataFrame(
                                                                to_numpy(self.dataset.table_dataset.x),
                                                                columns=self.feature_order
                                                            ),
                                                            y=self._get_structured_array(to_numpy(
                                                                self.dataset.table_dataset.y[task.name][:, 0]),
                                                                to_numpy(
                                                                    self.dataset.table_dataset.y[task.name][:, 1]
                                                                )),
                                                            predict_survival_function=wrapper)
            else:
                explainer = None
            survshap_explanation = survshap.ModelSurvSHAP(
                calculation_method='shap',
                function_type=function,
                random_state=11121
            )
            survshap_explanation.fit(explainer)
            self.survshap_explanation = survshap_explanation
            # print(survshap_explanation.full_result)
            # print(survshap_explanation.result)
            # colors = list(iter(plt.cm.rainbow(np.linspace(0, 1, len(survshap_explanation.individual_explanations)))))
            # for i, exp in enumerate(survshap_explanation.individual_explanations):
            #     # print(i.result.loc[0].iloc[6:])
            #     x = exp.timestamps
            #     for j in range(len(exp.result.iloc[:, 0])):
            #         y = [k for k in exp.result.loc[j].iloc[6:]]
            #         plt.plot(x, y, color=colors[i])
            # plt.show()
            # plt.close()
                # print(i.timestamps)
                # print([k for k in i.result.loc[0].iloc[6:]])
                # print([k for k in i.result.loc[1].iloc[6:]])
                # print([k for k in i.result.loc[2].iloc[6:]])
                # print([k for k in i.result.loc[3].iloc[6:]])
                # print([k for k in i.result.loc[4].iloc[6:]])
                # print([k for k in i.result.loc[5].iloc[6:]])
                # print(i.result.loc[1].iloc[6:])
                # print(i.result.loc[2].iloc[6:])
                # print(i.result.loc[3].iloc[6:])
                # print(i.result.loc[4].iloc[6:])
                # print(i.result.loc[5].iloc[6:])
        self.fitted += [task.name]
        return self.survshap_explanation

    def plot_shap_lines_for_all_patients(
            self,
            feature: Union[Dict[Tuple[str], List[int]], str],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
    ) -> None:
        """
        Plots the Shap values of a single feature as a function of time for all patients.

        Parameters
        ----------
        feature : Union[Dict[Tuple[str], List[int]], str]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {['PSA', 'AGE']: [0, 1, 2, 3, 4, 4], ['GLEASON_GLOBAL', 'GLEASON_SECONDARY']: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input the chosen feature.
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
        if isinstance(feature, str):
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                model_plot_shap_lines_for_all_individuals(
                    full_result=explanation.full_result,
                    timestamps=explanation.timestamps,
                    event_inds=explanation.event_ind,
                    event_times=explanation.event_times,
                    variable=feature
                )
        else:
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                for features, patients in feature.items():
                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]
                    colors = list(iter(plt.cm.rainbow(np.linspace(0, 1, len(patients)))))
                    for i, patient_index in enumerate(patients):
                        exp = explanation.individual_explanations[patient_index]
                        x = exp.timestamps
                        for feature_name in features:
                            y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            plt.plot(x, y, color=colors[i])
                    plt.show()
                    plt.close()

    def plot_shap_lines_for_features(
            self,
            features: Union[List[str], Dict[Tuple[str], List[int]]],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            discretise: Optional[Tuple[int, List[str]]] = None
    ) -> None:
        """
        Plots the Shap values of each desired feature as a function of time for all patients.

        Parameters
        ----------
        features : Union[List[str], Dict[Tuple[str], List[int]]]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {['PSA', 'AGE']: [0, 1, 2, 3, 4, 4], ['GLEASON_GLOBAL', 'GLEASON_SECONDARY']: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input a list of the chosen features.
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
        if isinstance(features, list):
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
        else:
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                cmap = list(iter(plt.cm.rainbow(np.linspace(0, 1, len(self.feature_order)))))
                color_dict = {feature: cmap[i] for i, feature in enumerate(self.feature_order)}
                for features_list, patients in features.items():
                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]
                    for i, patient_index in enumerate(patients):
                        exp = explanation.individual_explanations[patient_index]
                        x = exp.timestamps
                        for feature_name in features_list:
                            y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            plt.plot(x, y, color=color_dict[feature_name])
                    plt.show()
                    plt.close()

    def plot_shap_average_of_absolute_value(
            self,
            features: Union[List[str], Dict[Tuple[str], List[int]]],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]],
    ) -> None:
        """
        Plots the average of the absolute value of each Shap value as a function of time.

        Parameters
        ----------
        features : List[str]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {['PSA', 'AGE']: [0, 1, 2, 3, 4, 4], ['GLEASON_GLOBAL', 'GLEASON_SECONDARY']: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input a list of the chosen features.
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
        if isinstance(features, list):
            for task in tasks:
                explainer = self.compute_explanation(task=task, function=function)
                model_plot_mean_abs_shap_values(
                    result=explainer.result,
                    timestamps=explainer.timestamps,
                    event_inds=explainer.event_ind,
                    event_times=explainer.event_times,
                    variables=features
                )
        else:
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                cmap = list(iter(plt.cm.rainbow(np.linspace(0, 1, len(self.feature_order)))))
                color_dict = {feature: cmap[i] for i, feature in enumerate(self.feature_order)}
                for features_list, patients in features.items():
                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]
                    sum_of_values = {key: np.array([0 for _ in [explanation.timestamps]]) for key in self.feature_order}
                    for i, patient_index in enumerate(patients):
                        exp = explanation.individual_explanations[patient_index]
                        for feature_name in features_list:
                            y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            sum_of_values[feature_name] = sum_of_values.get(feature_name) + np.abs(np.array(y))
                    for feature_name in features_list:
                        average_values = sum_of_values[feature_name]/(1+len(patients))
                        plt.plot(explanation.timestamps, average_values, color=color_dict[feature_name])
                    plt.show()
                    plt.close()
