"""
    @file:              survshap_explainer.py
    @Author:            Félix Desroches

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file contains a class used to analyse and explain how a model works using shapley values
    for survival tasks.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sksurv.functions import StepFunction
import survshap
from survshap.model_explanations.plot import (
    model_plot_mean_abs_shap_values,
    model_plot_shap_lines_for_all_individuals,
    model_plot_shap_lines_for_variables
)
import torch

from applications.constants import *
from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset
from ..models.base.model import Model
from ..tasks.base import Task, TableTask
from ..tasks.containers import TaskList
from ..tasks.survival_analysis import SurvivalAnalysisTask
from ..tools.plot import terminate_figure
from ..tools.transforms import to_numpy


class StepFunctionWrapper(StepFunction):
    """
    Wrapper to convert the output of a StepFunction into numpy ndarray.
    """

    def __init__(self, step_func):
        super().__init__(x=step_func.x, y=step_func.y, a=step_func.a, b=step_func.b)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Wrapper of the call function to convert the output.
        """
        old_return = super().__call__(*args, **kwargs)
        if isinstance(old_return, np.ndarray):
            return old_return
        elif isinstance(old_return, torch.Tensor):
            return old_return.numpy()


class DataFrameWrapper:
    """
    Wrapper to create a way to compute the cumulative hazard functions and survival functions using pandas DataFrames.
    """
    def __init__(
            self,
            function,
            features: List[str],
            task: SurvivalAnalysisTask,
    ) -> None:
        """
        Sets up the required attributes.

        Parameters
        ----------
        function
            The function with which to compute the cumulative hazard or survival functions.
        features : List[str]
            The dataset with which to compute the SurvSHAP values.
        task : SurvivalAnalysisTask
            The task for which to compute the SurvSHAP values.
        """
        self.function = function
        self.task = task
        self.table_features_order = features

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
                for feature in self.table_features_order:
                    table_data[feature] = torch.unsqueeze(torch.tensor(data_frame.loc[:, feature]), -1)
            elif data_frame.ndim == 1:
                for feature in self.table_features_order:
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
            The converted FeaturesType.
        """
        array_list_to_cat = []
        for feature in self.table_features_order:

            datum = features_type.table[feature]
            if isinstance(datum, torch.Tensor):
                datum = to_numpy(datum)
            elif isinstance(datum, np.ndarray):
                pass
            else:
                datum = np.array([datum])
            datum = np.expand_dims(datum, 1)
            array_list_to_cat.append(datum)

        return pandas.DataFrame(np.concatenate(array_list_to_cat, 1), columns=self.table_features_order)

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
                for feature in self.table_features_order:
                    table_data[feature] = torch.unsqueeze(
                        torch.tensor(initial_array[:, self.table_features_order.index(feature)]),
                        -1
                    )
            elif initial_array.ndim == 1:
                for feature in self.table_features_order:
                    table_data[feature] = torch.unsqueeze(
                        torch.tensor(initial_array[self.table_features_order.index(feature)]),
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

        if isinstance(prediction[self.task.name], torch.Tensor) and prediction[self.task.name].get_device != -1:

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
            random_state: int = 11121,
            mask: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Sets up the required attributes.

        Parameters
        ----------
        model : Model
            The model to explain.
        dataset : ProstateCancerDataset
            The dataset to use to compute the SurvSHAP values.
        random_state : int
            A seed to set determinism. Defaults to 11121
        mask : Optional[Tuple[int, int]]
            A tuple of the first and last patients to limit the number of computational time required. Defaults to use
            all patients which can be long.
        """
        mpl.rc("axes", edgecolor="k")
        mpl.rcParams["mathtext.fontset"] = "cm"
        mpl.rcParams["font.family"] = "STIXGeneral"
        assert dataset.image_dataset is None and dataset.table_dataset is not None, (
            "SurvSHAP values require a table dataset and cannot be computed with a model that requires an image dataset"
        )
        self.model = model
        self.dataset = dataset
        self.predictions_dict = {k: to_numpy(v) for k, v in self.model.predict_on_dataset(dataset=self.dataset).items()}
        self.feature_order = self.dataset.table_dataset.features_columns
        self.fitted = {}
        self.random_state = random_state
        self.mask = mask

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
            (N, ) array with event indicator.
        event_time : np.ndarray
            (N, ) array with event time.

        Returns
        -------
        structured_array : np.ndarray
            (N, 2) structured array with event indicator and event time.
        """
        structured_array = np.empty(shape=(len(event_indicator),), dtype=[("event", int), ("time", float)])
        structured_array["event"] = event_indicator.astype(bool)
        structured_array["time"] = event_time

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
        assert function == "chf" or function == "sf", (
            "Only the survival function ('sf') and cumulative hazard function ('chz') are implemented."
        )

        if self.fitted.get((function, task.name), None) is not None:
            return self.fitted[(function, task.name)]
        else:
            y = self._get_structured_array(
                to_numpy(self.dataset.table_dataset.y[task.name][:, 0]),
                to_numpy(self.dataset.table_dataset.y[task.name][:, 1])
            )
            if self.mask is not None:
                y = y[self.mask[0]:self.mask[1]]
            data = pandas.DataFrame(to_numpy(self.dataset.table_dataset.x), columns=self.feature_order)
            if self.mask is not None:
                data = data.iloc[self.mask[0]:self.mask[1], :]
            if function == "chf":
                wrapper = DataFrameWrapper(
                    task.breslow_estimator.get_cumulative_hazard_function,
                    self.dataset.table_dataset.features_columns,
                    task
                )
                explainer = survshap.SurvivalModelExplainer(model=self.model,
                                                            data=data,
                                                            y=y,
                                                            predict_cumulative_hazard_function=wrapper)
            elif function == "sf":
                wrapper = DataFrameWrapper(
                    task.breslow_estimator.get_survival_function,
                    self.dataset.table_dataset.features_columns,
                    task
                )
                explainer = survshap.SurvivalModelExplainer(model=self.model,
                                                            data=data,
                                                            y=y,
                                                            predict_survival_function=wrapper)
            else:
                explainer = None

            survshap_explanation = survshap.ModelSurvSHAP(
                calculation_method="shap",
                function_type=function,
                random_state=self.random_state
            )
            survshap_explanation.fit(explainer)
            self.fitted[(function, task.name)] = survshap_explanation

            return survshap_explanation

    def plot_shap_lines_for_all_patients(
            self,
            features: Union[Dict[Tuple[str, ...], List[int]], str],
            function: str,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            normalize: bool = False,
            legend_length: int = 5,
            **kwargs
    ) -> None:
        """
        Plots the Shap values desired features as a function of time for all desired patients. Each color indicates a
        different patient, all SurvSHAP values for a patient are the same color.

        Parameters
        ----------
        features : Union[Dict[Tuple[str, ...], List[int]], str]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {["PSA", "AGE"]: [0, 1, 2, 3, 4], ["GLEASON_GLOBAL", "GLEASON_SECONDARY"]: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input the chosen feature.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        show : bool
            Whether to show the graph. Defaults to True.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        normalize : bool
            Whether to normalize the values by dividing by the sum of the absolute value of the SurvSHAP(t) values for
            a given patient at each timestamp. Defaults to False.
        legend_length : int
            The number of patients to show in the legend, too many can clutter and hide the graph. Defaults to 5
        """
        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )

        if isinstance(features, str):
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                model_plot_shap_lines_for_all_individuals(
                    full_result=explanation.full_result,
                    timestamps=explanation.timestamps,
                    event_inds=explanation.event_ind,
                    event_times=explanation.event_times,
                    variable=features
                )
        else:
            for task in tasks:
                fig, arr = plt.subplots(figsize=(8, 6))

                explanation = self.compute_explanation(task=task, function=function)

                for features_list, patients in features.items():
                    patch_list = []

                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]
                    if normalize:
                        sum_of_values = {
                            key: np.array([0 for _ in [explanation.timestamps]]) for key in patients
                        }

                        for patient_index in patients:
                            exp = explanation.individual_explanations[patient_index]

                            for feature_name in features_list:
                                y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                                sum_of_values[patient_index] = sum_of_values.get(patient_index) + np.abs(np.array(y))
                    else:
                        sum_of_values = {
                            key: np.array([1 for _ in [explanation.timestamps]]) for key in patients
                        }
                    colors = list(iter(BLUE_TO_SAND(np.linspace(0, 1, len(patients)))))
                    patient_ids_dict = self.dataset.table_dataset.row_idx_to_ids

                    for i, patient_index in enumerate(patients):
                        exp = explanation.individual_explanations[patient_index]
                        x = exp.timestamps
                        if len(patch_list) < legend_length:
                            patch_list += [mpl.patches.Patch(color=colors[i], label=patient_ids_dict[patient_index])]

                        for feature_name in features_list:
                            y = np.array(
                                [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            )/sum_of_values[patient_index]
                            arr.plot(x, y, color=colors[i], linewidth=2)

                    normalize_name = "normalized" if normalize else "not_normalized"
                    arr.set_xlabel(kwargs.get("xlabel", "Time $($months$)$"), fontsize=18)
                    arr.set_ylabel(kwargs.get("ylabel", "SHAP value"), fontsize=18)
                    arr.legend(handles=patch_list, edgecolor="k", fontsize=16, handlelength=1.5, loc="upper right")
                    arr.set_xlim(None, 190)
                    arr.minorticks_on()
                    arr.tick_params(axis="both", direction="in", color="k", which="major", labelsize=16, length=6)
                    arr.tick_params(axis="both", direction="in", color="k", which="minor", labelsize=16, length=3)
                    arr.grid(False)

                    if path_to_save_folder is not None:
                        path = os.path.join(
                            path_to_save_folder,
                            f"{task.target_column}_{function}_"
                            f"{normalize_name}_{kwargs.get('filename', 'SHAP_value_for_patients.pdf')}"
                        )
                    else:
                        path = None

                    terminate_figure(path_to_save=path, show=show, fig=fig, **kwargs)

    def plot_shap_lines_for_features(
            self,
            features: Union[List[str], Dict[Tuple[str, ...], List[int]]],
            function: str,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            tasks: Optional[Union[Task, TaskList, List[Task]]] = None,
            discretize: Optional[Tuple[int, List[str]]] = None,
            normalize: bool = False,
            **kwargs
    ) -> None:
        """
        Plots the Shap values of each desired feature as a function of time for all patients.

        Parameters
        ----------
        features : Union[List[str], Dict[Tuple[str, ...], List[int]]]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {["PSA", "AGE"]: [0, 1, 2, 3, 4], ["GLEASON_GLOBAL", "GLEASON_SECONDARY"]: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input a list of the chosen features.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        show : bool
            Whether to show the graph. Defaults to True.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        discretize : Optional[Tuple[int, List[str]]]
            If transforming a feature into a discretized state is desired, then it is the Tuple of the number of bins
            and a list of the features to discretize.
        normalize : bool
            Whether to normalize the values by dividing by the sum of the absolute value of the SurvSHAP(t) values for
            a given patient at each timestamp. Defaults to False.
        """
        if tasks is None:
            tasks = self.dataset.tasks.survival_analysis_tasks
        else:
            tasks = TaskList(tasks)
            assert all(isinstance(task, TableTask) for task in tasks), (
                f"All tasks must be instances of 'TableTask'."
            )

        if discretize is not None:
            method = "quantile"
        else:
            method = ""
            discretize = (5, [])

        if isinstance(features, list):
            for task in tasks:
                explanation = self.compute_explanation(task=task, function=function)
                model_plot_shap_lines_for_variables(
                    full_result=explanation.full_result,
                    timestamps=explanation.timestamps,
                    event_inds=explanation.event_ind,
                    event_times=explanation.event_times,
                    variables=features,
                    to_discretize=discretize[1],
                    discretization_method=method,
                    n_bins=discretize[0],
                    show=True
                )
        else:
            for task in tasks:
                fig, arr = plt.subplots(figsize=(8, 6))

                explanation = self.compute_explanation(task=task, function=function)
                # cmap = list(iter(BLUE_TO_SAND(np.linspace(0, 1, len(features.keys())))))
                # color_dict = {feature: cmap[i] for i, feature in enumerate(list(features.keys()))}

                for features_list, patients in features.items():
                    cmap = list(iter(BLUE_TO_SAND(np.linspace(0, 1, len(features_list)))))
                    color_dict = {feature: cmap[i] for i, feature in enumerate(features_list)}
                    patch_list = []

                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]

                    if normalize:
                        sum_of_values = {
                            key: np.array([0 for _ in [explanation.timestamps]]) for key in patients
                        }

                        for patient_index in patients:
                            exp = explanation.individual_explanations[patient_index]

                            for feature_name in features_list:
                                y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                                sum_of_values[patient_index] = sum_of_values.get(patient_index) + np.abs(np.array(y))
                    else:
                        sum_of_values = {
                            key: np.array([1 for _ in [explanation.timestamps]]) for key in patients
                        }

                    for patient_index in patients:
                        exp = explanation.individual_explanations[patient_index]
                        x = exp.timestamps

                        for feature_name in features_list:
                            y = np.array(
                                [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            )/sum_of_values[patient_index]
                            if feature_name in PN_TASK_FEATURES:
                                if isinstance(task, PN_TASK):
                                    arr.plot(x, y, color=LEGEND_NAMES_AND_COLORS[feature_name][1], linewidth=2,
                                         label=LEGEND_NAMES_AND_COLORS[feature_name][0], linestyle='dashed')
                            elif feature_name in BCR_TASK_FEATURES:
                                if isinstance(task, BCR_TASK):
                                    arr.plot(x, y, color=LEGEND_NAMES_AND_COLORS[feature_name][1], linewidth=2,
                                             label=LEGEND_NAMES_AND_COLORS[feature_name][0], linestyle='dashed')
                            else:
                                arr.plot(x, y, color=LEGEND_NAMES_AND_COLORS[feature_name][1], linewidth=2, label=LEGEND_NAMES_AND_COLORS[feature_name][0])

                    # for feature_name in features_list:
                    #     if feature_name in PN_TASK_FEATURES:
                    #         if isinstance(task, PN_TASK):
                    #             "f"
                    #     elif feature_name in BCR_TASK_FEATURES:
                    #         if isinstance(task, BCR_TASK):
                    #             "f"
                    #     else:
                    #         patch_list += [mpl.patches.Patch(
                    #             color=LEGEND_NAMES_AND_COLORS[feature_name][1],
                    #             label=LEGEND_NAMES_AND_COLORS[feature_name][0]
                    #         )]

                    normalize_name = "normalized" if normalize else "not_normalized"
                    arr.set_xlabel(kwargs.get("xlabel", "Time $($months$)$"), fontsize=18)
                    arr.set_ylabel(kwargs.get("ylabel", f"SHAP value"), fontsize=18)
                    arr.legend(edgecolor="k", fontsize=13, handlelength=1.5, loc="upper right")
                    arr.set_xlim(None, 190)
                    arr.minorticks_on()
                    arr.tick_params(axis="both", direction="in", color="k", which="major", labelsize=16, length=6)
                    arr.tick_params(axis="both", direction="in", color="k", which="minor", labelsize=16, length=3)
                    arr.grid(False)
                    arr.set_xlim(None, 190)

                    if path_to_save_folder is not None:
                        path = os.path.join(
                            path_to_save_folder,
                            f"{task.target_column}_{function}_{normalize_name}"
                            f"_{kwargs.get('filename', 'SHAP_value_for_features.pdf')}"
                        )
                    else:
                        path = None

                    terminate_figure(path_to_save=path, show=show, fig=fig, **kwargs)

    def plot_shap_average_of_absolute_value(
            self,
            features: Union[List[str], Dict[Tuple[str, ...], List[int]]],
            function: str,
            tasks: Optional[Union[Task, TaskList, List[Task]]],
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            normalize: bool = False,
            **kwargs

    ) -> None:
        """
        Plots the average of the absolute value of each Shap value as a function of time.

        Parameters
        ----------
        features : Union[List[str], Dict[Tuple[str, ...], List[int]]]
            A dictionary with keys being a tuple of desired features for a graph and values being a list of the indexes
            of the patients to put on the graph. The number of graphs will equal the number of keys in the dictionary.
            E.g. {["PSA", "AGE"]: [0, 1, 2, 3, 4], ["GLEASON_GLOBAL", "GLEASON_SECONDARY"]: [5, 6, 7, 8, 9]} would
            create two graphs. An empty list of patients indexes will graph all patients. If the original SurvSHAP(t)
            graphs are desired, simply input a list of the chosen features.
        function : str
            The function for which to compute the explanation, either "chf" for cumulative hazard function or "sf" for
            survival function.
        tasks : Optional[Union[Task, TaskList, List[Task]]]
            The tasks for which to plot the graphs. One graph will be created for each task.
        show : bool
            Whether to show the graph. Defaults to True.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        normalize : bool
            Whether to normalize the values by dividing by the sum of the absolute value of the SurvSHAP(t) values for
            a given patient at each timestamp. Defaults to False.
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
                explanation = self.compute_explanation(task=task, function=function)
                model_plot_mean_abs_shap_values(
                    result=explanation.result,
                    timestamps=explanation.timestamps,
                    event_inds=explanation.event_ind,
                    event_times=explanation.event_times,
                    variables=features
                )
        else:
            for task in tasks:
                fig, arr = plt.subplots(figsize=(8, 6))

                explanation = self.compute_explanation(task=task, function=function)
                # cmap = list(iter(BLUE_TO_SAND(np.linspace(0, 1, len(features.keys())))))
                # color_dict = {feature: cmap[i] for i, feature in enumerate(list(features.keys()))}

                for features_list, patients in features.items():
                    cmap = list(iter(BLUE_TO_SAND(np.linspace(0, 1, len(features_list)))))
                    color_dict = {feature: cmap[i] for i, feature in enumerate(features_list)}
                    if len(patients) == 0:
                        patients = [i for i in range(len(explanation.individual_explanations))]

                    sum_of_values = {key: np.array([0 for _ in [explanation.timestamps]]) for key in self.feature_order}

                    for patient_index in patients:
                        exp = explanation.individual_explanations[patient_index]

                        for feature_name in features_list:
                            y = [k for k in exp.result.loc[self.feature_order.index(feature_name), :].iloc[6:]]
                            sum_of_values[feature_name] = sum_of_values.get(feature_name) + np.abs(np.array(y))
                    patch_list = []

                    if normalize:
                        sum_of_averages = np.array([0 for _ in [explanation.timestamps]])
                        for feature_name in features_list:
                            sum_of_averages = sum_of_averages + sum_of_values[feature_name]/(len(patients))
                    else:
                        sum_of_averages = np.array([1 for _ in [explanation.timestamps]])

                    for feature_name in features_list:
                        average_values = sum_of_values[feature_name]/(len(patients))/sum_of_averages

                        arr.plot(explanation.timestamps, average_values, color=color_dict[feature_name], linewidth=2)
                        if feature_name in PN_TASK_FEATURES and task != PN_TASK:
                            continue
                        elif feature_name in BCR_TASK_FEATURES and task != BCR_TASK:
                            continue
                        else:
                            patch_list += [mpl.patches.Patch(color=color_dict[feature_name], label=feature_name)]

                    arr.set_xlabel(kwargs.get("xlabel", "Time $($months$)$"), fontsize=18)
                    arr.set_ylabel(kwargs.get("ylabel", "SHAP value"), fontsize=18)
                    arr.legend(handles=patch_list, edgecolor="k", fontsize=13, handlelength=1.5, loc="upper right")
                    arr.set_xlim(None, 190)
                    arr.minorticks_on()
                    arr.tick_params(axis="both", direction="in", color="k", which="major", labelsize=16, length=6)
                    arr.tick_params(axis="both", direction="in", color="k", which="minor", labelsize=16, length=3)
                    arr.grid(False)

                    if path_to_save_folder is not None:
                        path = os.path.join(
                            path_to_save_folder,
                            f"{task.target_column}_{function}"
                            f"_{kwargs.get('filename', 'average_of_absolute_value_of_SHAP.pdf')}"
                        )
                    else:
                        path = None

                    terminate_figure(path_to_save=path, show=show, fig=fig, **kwargs)
