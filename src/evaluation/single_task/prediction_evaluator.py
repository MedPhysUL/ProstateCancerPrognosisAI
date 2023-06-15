"""
    @file:              prediction_evaluator.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the user to gauge the
    quality of a model.
"""
import json
from typing import Dict, List, Optional, Union, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from torch import cat, float32, float64, tensor, Tensor

from ...data.datasets.prostate_cancer import FeaturesType, TargetsType
from ...metrics.single_task.base import Direction, MetricReduction
from ...tasks.base import Task
from ...tasks.containers.list import TaskList
from ...tools.transforms import to_numpy


class Output(NamedTuple):
    predictions: List
    targets: List


class PredictionEvaluator:
    def __init__(
            self,
            predictions: TargetsType,
            ground_truth: Union[dict, FeaturesType, Tensor],
            tasks: Union[Task, TaskList, List[Task]]
    ) -> None:
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        predictions : List[TargetsType]
            The predictions of the model from a dataset.
        ground_truth : List[Union[dict, FeaturesType, Tensor]]
            Ground truths to be used as a reference for the computation of the different metrics.
        tasks : Union[Task, TaskList, List[Task]]
            Object of the class TaskList that specifies for which tasks the model should be evaluated.
        """
        self.predictions_dict = predictions
        self.targets_dict = ground_truth
        self.tasks = TaskList(tasks)
        assert all(isinstance(task, Task) for task in self.tasks), (
            f"All tasks must be instances of 'TableTask'."
        )
        self.predictions = self._predictions_from_dataset()
        self.targets = self._targets_from_dataset()

    def _predictions_from_dataset(self) -> List[TargetsType]:
        """
        Generates predictions using a dataset, model and mask.

        Returns
        -------
        predictions : List[TargetsType]
            The predictions of the model on the dataset in a list.
        """
        predict_dict = self.predictions_dict
        dataset_length = len(predict_dict[list(predict_dict.keys())[0]])
        predict_list = [{} for _ in range(dataset_length)]
        for task in self.tasks.table_tasks:
            predictions = predict_dict.get(task.name).tolist()
            for i, prediction in enumerate(predictions):
                predict_list[i][task.name] = tensor(prediction)

        return predict_list

    def _targets_from_dataset(self) -> List[TargetsType]:
        """
        Returns the targets within a given dataset and mask.

        Returns
        ------
        targets : List[TargetsType]
            The targets in a dataset in a list.
        """
        targets_dict = self.targets_dict
        dataset_length = len(targets_dict[list(targets_dict.keys())[0]])
        targets_list = [{} for _ in range(dataset_length)]
        for task in self.tasks.table_tasks:
            targets = targets_dict.get(task.name).tolist()
            for i, target in enumerate(targets):
                if isinstance(target, int):
                    targets_list[i][task.name] = tensor([target])
                else:
                    targets_list[i][task.name] = tensor([target], dtype=float64)

        return targets_list

    @staticmethod
    def _update_segmentation_scores(
            seg_tasks: TaskList,
            segmentation_scores: Dict[str, Dict[str, list]],
            predictions: TargetsType,
            targets: TargetsType
    ) -> Dict[str, Dict[str, list]]:
        """
        Adds the data of one patient to a segmentation_scores dictionary.

        Parameters
        ----------
        seg_tasks : TaskList
            List of segmentation tasks for a dataset.
        segmentation_scores : Dict[str, Dict[str, list]]
            Score for each task for each metric for each patient.
        predictions : TargetsType
            The model's prediction for the patient.
        targets : TargetsType
            The target for the patient.

        Returns
        -------
        segmentation_scores : Dict[str, Dict[str, list]]
            Updated dictionary with the patient added.
        """
        for task in seg_tasks:
            for metric in task.unique_metrics:
                segmentation_scores[task.name][metric.name].append(
                    metric(predictions[task.name], targets[task.name], MetricReduction.NONE)
                )
        return segmentation_scores

    @staticmethod
    def _update_table_outputs(
            table_tasks: TaskList,
            table_outputs: Dict[str, Output],
            predictions: TargetsType,
            targets: TargetsType
    ) -> Dict[str, Output]:
        """
        Adds the data of one patient to a table_outputs dictionary.

        Parameters
        ----------
        table_tasks : TaskList
            List of table tasks for a dataset.
        table_outputs : Dict[str, Dict[str, list]]
            Score for each task for each metric for each patient.
        predictions : TargetsType
            The model's prediction for the patient.
        targets : TargetsType
            The target for the patient.

        Returns
        -------
        table_outputs : Dict[str, Dict[str, list]]
            Updated dictionary with the patient added.
        """
        for task in table_tasks:
            if task.metrics:
                table_outputs[task.name].predictions.append(predictions[task.name].item())
                table_outputs[task.name].targets.append(targets[task.name].tolist()[0])
        return table_outputs

    @staticmethod
    def _update_scores_with_segmentation_scores(
            seg_tasks: TaskList,
            segmentation_scores: Dict[str, Dict[str, list]],
            scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Adds the segmentation scores to the scores dictionary.

        Parameters
        ----------
        seg_tasks : TaskList
            List of segmentation tasks for a dataset.
        segmentation_scores : Dict[str, Dict[str, list]]
            Score for each task for each metric for each patient.
        scores : Dict[str, Dict[str, float]]
            Score for each task for each metric.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Updated dictionary with the segmentation scores added.
        """
        for task in seg_tasks:
            for metric in task.unique_metrics:
                scores[task.name][metric.name] = metric.perform_reduction(
                    tensor(segmentation_scores[task.name][metric.name], dtype=float32)
                )
        return scores

    @staticmethod
    def _update_scores_with_table_scores(
            table_tasks: TaskList,
            table_outputs: Dict[str, Output],
            scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
         Adds the table scores to the scores dictionary.

        Parameters
        ----------
        table_tasks : TaskList
            List of table tasks for a dataset.
        table_outputs : Dict[str, Dict[str, list]]
            Score for each task for each metric for each patient.
        scores : Dict[str, Dict[str, float]]
            Score for each task for each metric.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Updated dictionary with the table scores added.
        """
        for task in table_tasks:
            if task.metrics:
                output = table_outputs[task.name]
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric(to_numpy(output.predictions), to_numpy(output.targets))
        return scores

    @staticmethod
    def _update_outputs_dict(
            binary_classification_tasks: TaskList,
            outputs_dict: Dict[str, Output],
            predictions: TargetsType,
            targets: TargetsType
    ) -> Dict[str, Output]:
        """
        Creates the outputs dictionary containing the predictions and targets for a dataset.

        Parameters
        ----------
        binary_classification_tasks : TaskList
            List of the binary classification tasks in a dataset.
        outputs_dict : Dict[str, Output]
            Dictionary containing the predictions and targets for a dataset.
        predictions : TargetsType
            Prediction to be added to outputs_dict.
        targets : TargetsType
            Target to be added to outputs_dict.

        Returns
        -------
        outputs_dict : Dict[str, Output]
            outputs_dict with the patient added.
        """
        for task in binary_classification_tasks:
            outputs_dict[task.name].predictions.append(predictions[task.name].item())
            outputs_dict[task.name].targets.append(targets[task.name].item())
        return outputs_dict

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
            predictions, targets = self.predictions[mask], self.targets[mask]
        elif isinstance(mask, list):
            predictions, targets = [], []
            for i in mask:
                predictions.append(self.predictions[i])
                targets.append(self.targets[i])
        else:
            predictions, targets = self.predictions, self.targets

        for predictions, targets in tuple(zip(predictions, targets)):

            table_outputs = self._update_table_outputs(
                table_tasks=table_tasks,
                table_outputs=table_outputs,
                predictions=predictions,
                targets=targets
            )

        scores = self._update_scores_with_table_scores(
            table_tasks=table_tasks,
            table_outputs=table_outputs,
            scores=scores
        )

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
            predictions, targets = self.predictions[mask], self.targets[mask]
        elif isinstance(mask, list):
            predictions, targets = [], []
            for i in mask:
                predictions.append(self.predictions[i])
                targets.append(self.targets[i])
        else:
            predictions, targets = self.predictions, self.targets

        for predictions, targets in tuple(zip(predictions, targets)):

            outputs_dict = self._update_outputs_dict(
                binary_classification_tasks=binary_classification_tasks,
                outputs_dict=outputs_dict,
                predictions=predictions,
                targets=targets
            )

        self._fix_metric_threshold(
            binary_classification_tasks=binary_classification_tasks,
            outputs_dict=outputs_dict,
            thresholds=thresholds
        )

    @staticmethod
    def _terminate_figure(
            fig: plt.Figure,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Terminates current figure.

        Parameters
        ----------
        save : Optional[str]
            Path to save the figure.
        show : bool
            Whether to show figure.
        fig : plt.Figure
            Current figure.
        """
        fig.tight_layout()

        if save is not None:
            plt.savefig(save, **kwargs)
        if show:
            plt.show()
        plt.close(fig)

    def compute_metrics(
            self,
            save: Optional[str] = None,
            mask: Optional[Union[List[int], slice]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        save : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.
        mask : Optional[Union[List[int], slice]]
            Mask used to specify which patients to use when computing metrics. Uses all patients by default.
        """
        scores = self._compute_prediction_score(mask=mask)

        if save is not None:
            with open(f'{save}/metrics.json', 'w') as file_path:
                json.dump(scores, file_path)
        return scores

    def plot_classification_task_curves(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the different graphs for classification task metrics that can be visualised in a 2D graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        self.plot_confusion_matrix(show, save, **kwargs)
        self.plot_calibration_curve(show, save, **kwargs)
        self.plot_roc_curve(show, save, **kwargs)
        self.plot_precision_recall_curve(show, save, **kwargs)

    def plot_survival_analysis_curves(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the different graphs for survival analysis metrics that can be visualised in a 2D graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        self.plot_unique_times(show, save, **kwargs)
        self.plot_cum_baseline_hazard(show, save, **kwargs)
        self.plot_baseline_survival(show, save, **kwargs)
        self.plot_cum_hazard_function(show, save, **kwargs)
        self.plot_survival_function(show, save, **kwargs)

    def plot_unique_times(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow unique times graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            arr.plot(task.breslow_estimator.unique_times_)
            if save is not None:
                path = f'{save}/{task.name}_breslow_unique_times.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_cum_baseline_hazard(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow cumulative baseline hazard graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            cum_baseline_hazard = task.breslow_estimator.cum_baseline_hazard_
            arr.plot(cum_baseline_hazard.x, cum_baseline_hazard.y)

            if save is not None:
                path = f'{save}/{task.name}_breslow_cum_baseline_hazard.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_baseline_survival(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow baseline survival graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        for task in self.tasks.survival_analysis_tasks:
            fig, arr = plt.subplots()
            baseline_survival = task.breslow_estimator.baseline_survival_
            arr.plot(baseline_survival.x, baseline_survival.y)

            if save is not None:
                path = f'{save}/{task.name}_breslow_baseline_survival.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_cum_hazard_function(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow cumulative hazard function graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        for task in self.tasks.survival_analysis_tasks:
            prediction = {}
            fig, arr = plt.subplots()
            for prediction_element in self.predictions:
                if prediction.get(task.name, None) is not None:
                    prediction[task.name] = cat(
                        (prediction.get(task.name), prediction_element[task.name]),
                        dim=-1
                    )
                else:
                    prediction[task.name] = (prediction_element[task.name])
            for chf_func in task.breslow_estimator.get_cumulative_hazard_function(prediction[task.name]):
                arr.step(chf_func.x, chf_func(chf_func.x), where="post")

            if save is not None:
                path = f'{save}/{task.name}_breslow_cum_hazard_function.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_survival_function(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the breslow survival function graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig.
        """
        for task in self.tasks.survival_analysis_tasks:
            prediction = {}
            fig, arr = plt.subplots()
            for prediction_element in self.predictions:
                if prediction.get(task.name, None) is not None:
                    prediction[task.name] = cat(
                        (prediction.get(task.name), prediction_element[task.name]),
                        dim=-1
                    )
                else:
                    prediction[task.name] = (prediction_element[task.name])
            for survival_func in task.breslow_estimator.get_survival_function(prediction[task.name]):
                arr.step(survival_func.x, survival_func(survival_func.x), where="post")

            if save is not None:
                path = f'{save}/{task.name}_breslow_survival_function.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_confusion_matrix(
            self,
            show: bool,
            save: Optional[str] = None,
            threshold: Optional[Union[int, List[int], slice]] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        threshold : Optional[Union[int, List[int], slice]]
            Either the threshold or a mask describing the patients to use when optimising the threshold to use when
            computing binary classification from continuous probability. If no values are given, then the threshold is
            computed using all patients.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.confusion_matrix.

        """
        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            if not isinstance(threshold, int):
                self._fix_thresholds_to_optimal_values(mask=threshold)
                threshold = task.decision_threshold_metric.threshold
            y_true, y_pred = self.targets_dict[task.name], []
            for predictions in self.predictions:
                y_pred.append(1) if predictions[task.name][0] >= threshold else y_pred.append(0)

            arr.imshow(confusion_matrix(
                y_true,
                y_pred,
                labels=kwargs.get('labels', None),
                sample_weight=kwargs.get('sample_weight', None),
                normalize=kwargs.get('normalize', None)
            ))

            if save is not None:
                path = f'{save}/{task.name}_confusion_matrix.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_calibration_curve(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.calibration.calibration_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true, y_pred = self.targets_dict[task.name], self.predictions_dict[task.name]
            prob_true, prob_pred = calibration_curve(
                y_true,
                y_pred,
                pos_label=kwargs.get('pos_label', None),
                normalize=kwargs.get('normalize', 'deprecated'),
                n_bins=kwargs.get('n_bins', 5),
                strategy=kwargs.get('strategy', 'uniform')
                )
            arr.plot(prob_true, prob_pred, 'go')
            arr.plot([1, 0], [1, 0], 'k')

            if save is not None:
                path = f'{save}/{task.name}_calibration_curve.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_roc_curve(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.roc_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true, y_pred = self.targets_dict[task.name], self.predictions_dict[task.name]
            fpr, tpr, threshold = roc_curve(
                y_true,
                y_pred,
                pos_label=kwargs.get('pos_label', None),
                sample_weight=kwargs.get('sample_weight', None),
                drop_intermediate=kwargs.get('drop_intermediate', True)
            )
            arr.plot(fpr, tpr, 'g')
            arr.plot([1, 0], [1, 0], 'k')

            if save is not None:
                path = f'{save}/{task.name}_roc_curve.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)

    def plot_precision_recall_curve(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.precision_recall_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            fig, arr = plt.subplots()
            y_true, y_pred = self.targets_dict[task.name], self.predictions_dict[task.name]
            precision, recall, threshold = precision_recall_curve(
                y_true,
                y_pred,
                pos_label=kwargs.get('pos_label', None),
                sample_weight=kwargs.get('sample_weight', None)
            )
            arr.step(recall, precision, 'g')

            if save is not None:
                path = f'{save}/{task.name}_precision_recall_curve.pdf'
            else:
                path = None
            self._terminate_figure(save=path, show=show, fig=fig, **kwargs)
