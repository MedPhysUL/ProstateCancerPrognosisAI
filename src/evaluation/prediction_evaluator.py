"""
    @file:              prediction_evaluator.py
    @Author:            Félix Desroches

    @Creation Date:     06/2023
    @Last modification: 07/2023

    @Description:       This file contains a class used to show metrics and graphs for the user to gauge the
    quality of a model.
"""

import json
import os
from typing import Dict, List, Optional, Union, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..data.datasets.prostate_cancer import TargetsType
from ..metrics.single_task.base import Direction
from ..tasks.base import TableTask, Task
from ..tasks.containers.list import TaskList
from ..tools.plot import terminate_figure
from ..tools.transforms import to_numpy


class Output(NamedTuple):
    """
    Class used to store the predictions and the targets of a model.

    Elements
    --------
    predictions : List[float]
        The predictions of the model from a dataset.
    targets : List[float]
        Ground truths to be used as a reference for the computation of the different metrics.
    """
    predictions: List[float]
    targets: List[float]


class PredictionEvaluator:
    """
    Class used to show metrics and graphs for the user to gauge the quality of a model. Can compute metrics for
    table tasks only.
    """

    def __init__(
            self,
            predictions: TargetsType,
            targets: TargetsType,
            tasks: Union[Task, TaskList, List[Task]],
            breslow_mask: List[int],
            fit_breslow_estimators: bool = True
    ) -> None:
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        predictions : TargetsType
            The predictions of the model from a dataset.
        targets : TargetsType
            Ground truths to be used as a reference for the computation of the different metrics.
        tasks : Union[Task, TaskList, List[Task]]
            Object of the class TaskList that specifies for which tasks the model should be evaluated.
        breslow_mask : List[int]
            Mask used to fit the breslow estimators, usually this is the train mask.
        fit_breslow_estimators : bool
            Whether to fit the breslow estimators, defaults to True.
        """
        self.predictions_dict = {k: to_numpy(v) for k, v in predictions.items()}
        self.targets_dict = {k: to_numpy(v) for k, v in targets.items()}

        if fit_breslow_estimators:
            self._fit_breslow_estimators(breslow_mask)

        self.tasks = TaskList(tasks)
        assert all(isinstance(task, TableTask) for task in self.tasks), (
            f"All tasks must be instances of 'TableTask'."
        )

        self.predictions_list = self.slice_patient_dictionary(self.predictions_dict, separate_patients=True)
        self.targets_list = self.slice_patient_dictionary(self.targets_dict, separate_patients=True)

    def _fit_breslow_estimators(
            self,
            breslow_mask: List[int]
    ) -> None:
        """
        Fit all survival analysis tasks' breslow estimators given the training dataset.

        Parameters
        ----------
        breslow_mask : List[int]
            Mask used to fit the breslow estimators, usually this is the train mask.
        """
        predictions = self.slice_patient_dictionary(self.predictions_dict, breslow_mask, None, False)
        targets = self.slice_patient_dictionary(self.targets_dict, breslow_mask, None, False)

        for task in self.tasks:
            pred, target = to_numpy(predictions[task.name]), to_numpy(targets[task.name])

            nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(target)
            if len(nonmissing_targets_idx) > 0:
                pred, target = pred[nonmissing_targets_idx], target[nonmissing_targets_idx]
                task.breslow_estimator.fit(pred[:, 0], target[:, 0], target[:, 1])

    @staticmethod
    def slice_patient_dictionary(
            patient_dict: TargetsType,
            patient_indexes: Optional[Union[int, List[int]]] = None,
            task_keys: Optional[Union[str, List[str]]] = None,
            separate_patients: bool = False
    ) -> Union[TargetsType, List[TargetsType]]:
        """
        Slices a patient dictionary by keeping only the desired patients and tasks.

        Parameters
        ----------
        patient_dict : TargetsType
            Dictionary containing all patients. The values associated with each task need to be of the same size.
        patient_indexes : Optional[Union[int, List[int]]]
            Either the index of the desired patients or a list of indexes. If no value are given, all patients are kept.
        task_keys : Optional[Union[str, List[str]]]
            Either the name of the desired task or a list of names. If no value are given, all tasks are kept.
        separate_patients : bool
            Whether to split the dictionary into a list of patient dictionaries. Is False by default.

        Returns
        -------
        modified input : Union[TargetsType, List[TargetsType]]
            Returns the input as either a shortened dictionary or a list of dictionaries.
        """
        if isinstance(patient_indexes, int):
            patient_indexes = [patient_indexes]
        if isinstance(task_keys, str):
            task_keys = [task_keys]
        modified_dict = {}

        if task_keys is not None:
            for task, values in patient_dict.items():
                if task in task_keys:
                    modified_dict[task] = values
        else:
            modified_dict = patient_dict

        if patient_indexes is not None:
            for task, values in modified_dict.items():
                values_list = values.tolist()
                desired_patient_values = []
                for i, value in enumerate(values_list):
                    if i in patient_indexes:
                        desired_patient_values.append(value)
                modified_dict[task] = np.array(desired_patient_values)

        if separate_patients:
            patients_list = [{} for _ in modified_dict[list(modified_dict.keys())[0]]]
            for task, values in modified_dict.items():
                for i, value in enumerate(values.tolist()):
                    patients_list[i][task] = np.array([value])
            return patients_list

        return modified_dict

    @staticmethod
    def _fix_metric_threshold(
            binary_classification_tasks: TaskList,
            outputs_dict: Dict[str, Output],
            thresholds: np.ndarray
    ) -> None:
        """
        Fixes the threshold value to be optimized for each task.

        Parameters
        ----------
        binary_classification_tasks : TaskList
            List of the binary classification tasks in a dataset.
        outputs_dict : Dict[str, Output]
            Dictionary containing all outputs and targets for a dataset.
        thresholds : np.ndarray
            Array of thresholds to try when looking for the best one.
        """
        for task in binary_classification_tasks:
            output = outputs_dict[task.name]

            for metric in task.metrics:
                scores = [metric(to_numpy(output.predictions), to_numpy(output.targets), t) for t in thresholds]

                if metric.direction == Direction.MINIMIZE:
                    metric.threshold = thresholds[np.argmin(scores)]
                elif metric.direction == Direction.MAXIMIZE:
                    metric.threshold = thresholds[np.argmax(scores)]

            for metric in task.metrics:
                if metric.direction == Direction.NONE:
                    metric.threshold = task.decision_threshold_metric.threshold

    def _compute_prediction_score(
            self,
            mask: Optional[Union[List[int], slice]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples using predictions and targets. Can't compute metrics for segmentation tasks.

        Parameters
        ----------
        mask : Optional[Union[List[int], slice]]
            Mask used to specify which patients to use when computing metrics. Uses all patients by default.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        table_tasks = self.tasks.table_tasks

        scores = {task.name: {} for task in self.tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}

        if isinstance(mask, slice):
            predictions, targets = self.predictions_list[mask], self.targets_list[mask]
        elif isinstance(mask, list):
            predictions = self.slice_patient_dictionary(self.predictions_dict, mask, None, True)
            targets = self.slice_patient_dictionary(self.targets_dict, mask, None, True)
        else:
            predictions, targets = self.predictions_list, self.targets_list

        for predictions, targets in tuple(zip(predictions, targets)):

            for task in table_tasks:
                if task.metrics:
                    table_outputs[task.name].predictions.append(predictions[task.name].item())
                    table_outputs[task.name].targets.append(targets[task.name].tolist()[0])

        for task in table_tasks:
            if task.metrics:
                output = table_outputs[task.name]
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric(to_numpy(output.predictions), to_numpy(output.targets))

        return scores

    def _fix_thresholds_to_optimal_values(
            self,
            mask: Optional[Union[List[int], slice]] = None
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        mask : Optional[Union[List[int], slice]]
            Mask used to specify which patients to use when optimizing the thresholds. Uses all patients by default.
        """
        binary_classification_tasks = self.tasks.binary_classification_tasks
        outputs_dict = {task.name: Output(predictions=[], targets=[]) for task in binary_classification_tasks}

        thresholds = np.linspace(start=0.01, stop=0.95, num=95)
        if isinstance(mask, slice):
            predictions, targets = self.predictions_list[mask], self.targets_list[mask]
        elif isinstance(mask, list):
            predictions = self.slice_patient_dictionary(self.predictions_dict, mask, None, True)
            targets = self.slice_patient_dictionary(self.targets_dict, mask, None, True)
        else:
            predictions, targets = self.predictions_list, self.targets_list

        for predictions, targets in tuple(zip(predictions, targets)):

            for task in binary_classification_tasks:
                outputs_dict[task.name].predictions.append(predictions[task.name].item())
                outputs_dict[task.name].targets.append(targets[task.name].item())

        self._fix_metric_threshold(
            binary_classification_tasks=binary_classification_tasks,
            outputs_dict=outputs_dict,
            thresholds=thresholds
        )

    def compute_score(
            self,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[Union[List[int], slice]] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        path_to_save_folder : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.
        mask : Optional[Union[List[int], slice]]
            Mask used to specify which patients to use when computing metrics. Uses all patients by default.
        """
        scores = self._compute_prediction_score(mask=mask)

        if path_to_save_folder is not None:
            path = os.path.join(path_to_save_folder, kwargs.get("filename", "metrics.json"))
            with open(path, "w") as file_path:
                json.dump(scores, file_path)
        return scores

    def plot_binary_classification_task_curves(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the different graphs for classification task metrics that can be visualised in a 2D graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        assert len(self.tasks.binary_classification_tasks) > 0, (
            "There needs to be at least one BinaryClassificationTask to plot binary classification task curves."
        )

        self.plot_confusion_matrix(show, path_to_save_folder, mask=mask, **kwargs)
        self.plot_calibration_curve(show, path_to_save_folder, mask=mask, **kwargs)
        self.plot_roc_curve(show, path_to_save_folder, mask=mask, **kwargs)
        self.plot_precision_recall_curve(show, path_to_save_folder, mask=mask, **kwargs)

    def plot_survival_analysis_task_curves(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the different graphs for survival analysis metrics that can be visualised in a 2D graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot survival analysis task curves."
        )

        self.plot_unique_times(show, path_to_save_folder, **kwargs)
        self.plot_cum_baseline_hazard(show, path_to_save_folder, **kwargs)
        self.plot_baseline_survival(show, path_to_save_folder, **kwargs)
        self.plot_cum_hazard_function(show, path_to_save_folder, mask=mask, **kwargs)
        self.plot_survival_function(show, path_to_save_folder, mask=mask, **kwargs)

    def plot_unique_times(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow unique times graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot a unique times graph."
        )

        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            unique_times = task.breslow_estimator.unique_times_
            time_list = [i for i in range(1, int(max(unique_times.tolist())))]
            event_list = [(sum(1 for time in unique_times.tolist() if time <= event_time)) for event_time in time_list]

            arr.plot(time_list, event_list)
            arr.set_xlabel(kwargs.get("xlabel", f"Time"))
            arr.set_ylabel(kwargs.get("ylabel", f"Cumulative events"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Unique Times"))
            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'breslow_unique_times.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_cum_baseline_hazard(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow cumulative baseline hazard graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot a cumulative baseline hazard graph."
        )

        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            cum_baseline_hazard = task.breslow_estimator.cum_baseline_hazard_
            arr.plot(cum_baseline_hazard.x, cum_baseline_hazard.y)
            arr.set_xlabel(kwargs.get("xlabel", f"Time"))
            arr.set_ylabel(kwargs.get("ylabel", f"Hazard"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Cumulative Baseline Hazard"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'cumulative_baseline_hazard.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_baseline_survival(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow baseline survival graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot a baseline survival graph."
        )

        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            baseline_survival = task.breslow_estimator.baseline_survival_
            arr.plot(baseline_survival.x, baseline_survival.y)
            arr.set_xlabel(kwargs.get("xlabel", f"Time"))
            arr.set_ylabel(kwargs.get("ylabel", f"Probability of survival"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Baseline Survival"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'baseline_survival.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_cum_hazard_function(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow cumulative hazard function graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        prediction = self.slice_patient_dictionary(
            patient_dict=self.predictions_dict,
            patient_indexes=mask
        )
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot a cumulative hazard function graph."
        )

        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            for chf_func in task.breslow_estimator.get_cumulative_hazard_function(prediction[task.name]):
                arr.step(chf_func.x, chf_func(chf_func.x), where="post")
            arr.set_xlabel(kwargs.get("xlabel", f"Time"))
            arr.set_ylabel(kwargs.get("ylabel", f"Probability"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Cumulative Hazard Function"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'cumulative_hazard_function.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_survival_function(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow survival function graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        prediction = self.slice_patient_dictionary(patient_dict=self.predictions_dict, patient_indexes=mask)
        assert len(self.tasks.survival_analysis_tasks) > 0, (
            "There needs to be at least one SurvivalAnalysisTask to plot a survival function graph."
        )

        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            for survival_func in task.breslow_estimator.get_survival_function(prediction[task.name]):
                arr.step(survival_func.x, survival_func(survival_func.x), where="post")
            arr.set_xlabel(kwargs.get("xlabel", f"Time"))
            arr.set_ylabel(kwargs.get("ylabel", f"Probability"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Survival Function"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'survival_function.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_confusion_matrix(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            threshold: Optional[Union[int, List[int], slice]] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        threshold : Optional[Union[int, List[int], slice]]
            Either the threshold or a mask describing the patients to use when optimising the threshold to use when
            computing binary classification from continuous probability. If no values are given, then the threshold is
            computed using all patients.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.confusion_matrix.
        """
        if not isinstance(threshold, int):
            self._fix_thresholds_to_optimal_values(mask=threshold)

        self._create_confusion_matrix_from_thresholds(
            show=show,
            path_to_save_folder=path_to_save_folder,
            threshold=threshold,
            mask=mask,
            **kwargs
        )

    def _create_confusion_matrix_from_thresholds(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            threshold: Optional[Union[int, List[int], slice]] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        threshold : Optional[Union[int, List[int], slice]]
            Either the threshold or a mask describing the patients to use when optimising the threshold to use when
            computing binary classification from continuous probability. If no values are given, then the threshold is
            computed using all patients.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.confusion_matrix.
        """
        assert len(self.tasks.binary_classification_tasks) > 0, (
            "There needs to be at least one BinaryClassificationTask to plot a confusion matrix."
        )

        print(len(self.tasks.binary_classification_tasks))
        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            if not isinstance(threshold, int):
                threshold = task.decision_threshold_metric.threshold
            y_true = self.slice_patient_dictionary(self.targets_dict, patient_indexes=mask)[task.name]
            y_pred = np.where(
                self.slice_patient_dictionary(self.predictions_dict, patient_indexes=mask)[task.name]
                >= threshold, 1, 0
            )
            matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=kwargs.get("labels", None),
                sample_weight=kwargs.get("sample_weight", None),
                normalize=kwargs.get("normalize", None)
            )

            sns.heatmap(matrix, cmap="gist_gray", annot=True, fmt="g")
            arr.set_title(kwargs.get("title", f"{task.target_column}: Confusion Matrix"))
            arr.set_xlabel(kwargs.get("xlabel", "Predictions"))
            arr.set_ylabel(kwargs.get("ylabel", "Ground Truth"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'confusion_matrix.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_calibration_curve(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            normalize: bool = True,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the calibration curve graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        normalize : bool
            Whether to normalize the prediction probability, defaults to True.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.calibration.calibration_curve.
        """
        assert len(self.tasks.binary_classification_tasks) > 0, (
            "There needs to be at least one BinaryClassificationTask to plot a calibration curve graph."
        )

        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true = self.slice_patient_dictionary(self.targets_dict, patient_indexes=mask)[task.name]
            y_prob = self.slice_patient_dictionary(self.predictions_dict, patient_indexes=mask)[task.name]
            if normalize:
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            prob_true, prob_pred = calibration_curve(
                y_true=y_true,
                y_prob=y_prob,
                pos_label=kwargs.get("pos_label", None),
                n_bins=kwargs.get("n_bins", 10),
                strategy=kwargs.get("strategy", "quantile")
                )
            arr.plot(prob_pred, prob_true, "bo")
            arr.plot(prob_pred, prob_true)
            arr.plot([1, 0], [1, 0], "k")
            arr.set_xlabel(kwargs.get("xlabel", f"Predicted probability"))
            arr.set_ylabel(kwargs.get("ylabel", f"Fraction of positives"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Calibration Curve"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'calibration_curve.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_roc_curve(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the ROC curve.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.roc_curve.
        """
        assert len(self.tasks.binary_classification_tasks) > 0, (
            "There needs to be at least one BinaryClassificationTask to plot a roc curve graph."
        )

        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true = self.slice_patient_dictionary(self.targets_dict, patient_indexes=mask)[task.name]
            y_pred = self.slice_patient_dictionary(self.predictions_dict, patient_indexes=mask)[task.name]
            fpr, tpr, threshold = roc_curve(
                y_true,
                y_pred,
                pos_label=kwargs.get("pos_label", None),
                sample_weight=kwargs.get("sample_weight", None),
                drop_intermediate=kwargs.get("drop_intermediate", True)
            )
            arr.plot(fpr, tpr, "g")
            arr.plot([1, 0], [1, 0], "k")
            arr.set_xlabel(kwargs.get("xlabel", f"False positive rate"))
            arr.set_ylabel(kwargs.get("ylabel", f"True positive rate"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: ROC Curve"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'roc_curve.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

    def plot_precision_recall_curve(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            mask: Optional[List[int]] = None,
            **kwargs
    ) -> None:
        """
        Creates the precision recall curve.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        path_to_save_folder : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        mask : Optional[List[int]]
            A mask to select which patients to use. If a subset was given, then the patient's ID refers to the position
            within the subset and not the original dataset. If no mask is given, all patients are used.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.precision_recall_curve.
        """
        assert len(self.tasks.binary_classification_tasks) > 0, (
            "There needs to be at least one BinaryClassificationTask to plot a precision recall curve graph."
        )

        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true = self.slice_patient_dictionary(self.targets_dict, patient_indexes=mask)[task.name]
            y_pred = self.slice_patient_dictionary(self.predictions_dict, patient_indexes=mask)[task.name]
            precision, recall, threshold = precision_recall_curve(
                y_true,
                y_pred,
                pos_label=kwargs.get("pos_label", None),
                sample_weight=kwargs.get("sample_weight", None)
            )
            arr.step(recall, precision, "g")
            arr.set_xlabel(kwargs.get("xlabel", f"Recall"))
            arr.set_ylabel(kwargs.get("ylabel", f"Precision"))
            arr.set_title(kwargs.get("title", f"{task.target_column}: Precision Recall curve"))

            if path_to_save_folder is not None:
                path = os.path.join(
                    path_to_save_folder,
                    f"{task.target_column}_{kwargs.get('filename', 'precision_recall_curve.pdf')}"
                )
            else:
                path = None
            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)
