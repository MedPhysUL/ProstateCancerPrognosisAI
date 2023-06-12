"""
    @file:              prediction_evaluator.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the human user to gauge the
    quality of a model.
"""
import json
import matplotlib.pyplot as plt
import torch

from typing import Any, Dict, List, NamedTuple, Optional, Union

from sklearn.calibration import calibration_curve

from src.data.datasets.prostate_cancer import FeaturesType
from src.tasks.containers.list import TaskList
from src.models.torch.base.torch_model import Output
from src.metrics.single_task.base import MetricReduction
from src.tools.transforms import to_numpy

from torch import float32, tensor, Tensor
from sklearn.metrics import confusion_matrix


class GraphConfig(NamedTuple):
    """
    Inputs for breslow graphs.

    Elements
    --------
    show : bool
        Whether to show the graph.
    save : Union[bool, str]
        Whether to save the graph, if so, then this value is the path to the save folder.
    kwargs : Dict[str, Any]
        Dict of positional arguments to pass on to matplotlib.pyplot.savefig(), use {} if there are none.
    """
    show: bool
    save: Union[bool, str]
    kwargs: Dict[str, Any]


class TaskInput(NamedTuple):
    """
    Inputs for breslow graphs.

    Elements
    --------
    task_name : str
        Name of the current task, used to create an unambiguous file name.
    show : bool
        Whether to show the graph.
    save : Union[bool, str]
        Whether to save the graph, if so, then this value is the path to the save folder.
    kwargs : Dict[str, Any]
        Dict of positional arguments to pass on to matplotlib.pyplot.savefig(), use {} if there are none.
    """
    task_name: str
    show: bool
    save: Union[bool, str]
    kwargs: Dict[str, Any]


class PredictionEvaluator:
    def __init__(self,
                 predictions,
                 ground_truth: List[Union[dict, FeaturesType, Tensor]],
                 tasks: TaskList
                 ):
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        predictions
            Either the dataset with which the evaluation is desired or the predictions of the model from a dataset.
        ground_truth : Optional[List[Union[dict, FeaturesType, Tensor]]]
            Ground truths to be used as a reference for the computation of the different metrics. This argument is
            required if predictions are used.
        tasks : Optional[Tasklist]
            Object of the class TaskList that specifies for which tasks the model should be evaluated. This argument
            is required if predictions are used.
        """

        self.predictions = predictions
        self.ground_truth = ground_truth
        self.tasks = tasks

    def _score_on_predictions(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples using predictions and ground truths.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        table_tasks, seg_tasks = self.tasks.table_tasks, self.tasks.segmentation_tasks

        scores = {task.name: {} for task in self.tasks}
        segmentation_scores = {task.name: {metric.name: [] for metric in task.unique_metrics} for task in seg_tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}
        for predictions, targets in tuple(zip(self.predictions, self.ground_truth)):
            for task in seg_tasks:
                for metric in task.unique_metrics:
                    segmentation_scores[task.name][metric.name].append(
                        metric(predictions[task.name], targets[task.name], MetricReduction.NONE)
                    )

            for task in table_tasks:
                if task.metrics:
                    table_outputs[task.name].predictions.append(predictions[task.name].item())
                    table_outputs[task.name].targets.append(targets[task.name].tolist()[0])

        for task in seg_tasks:
            for metric in task.unique_metrics:
                scores[task.name][metric.name] = metric.perform_reduction(
                    tensor(segmentation_scores[task.name][metric.name], dtype=float32)
                )

        for task in table_tasks:
            if task.metrics:
                output = table_outputs[task.name]
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric(to_numpy(output.predictions), to_numpy(output.targets))

        return scores

    def scalar_metrics(self,
                       metrics: Optional[Union[str, List[str]]] = None,
                       return_metrics: bool = True,
                       save: Union[bool, str] = False
                       ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        metrics : Optional[Union[str, List[str]]]
            Either the metric to compute, a list of the metrics or "all" which will compute all metrics. Defaults to
            None which computes no metric.
        return_metrics : bool
            Whether to return the computed metrics. False means that the metric are computed but not returned. Defaults
            to True which returns the metrics as a dictionary.
        save : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.
        """
        scalar_metrics = {}
        if metrics is None:
            metrics = []
        elif metrics == "all":
            metrics = [
                       'BinaryAccuracy',
                       'BinaryBalancedAccuracy',
                       'Sensitivity',
                       'Specificity',
                       'ConcordanceIndexCensored',
                       'AUC',
                       'Dice'
                       ]
        elif isinstance(metrics, str):
            metrics = [metrics]

        scores = self._score_on_predictions()
        for task, metric_dict in scores.items():
            scalar_metrics[task] = {}
            metric_present = False
            for metric, value in metric_dict.items():
                if metric in metrics:
                    metric_present = True
                    scalar_metrics[task][metric] = value
            if not metric_present:
                scalar_metrics[task] = "N/A"

        if save:
            path = save + '/scalar_metrics.json'
            with open(path, 'w') as file_path:
                json.dump(scalar_metrics, file_path)
        if return_metrics:
            return scalar_metrics

    def visual_metrics(self,
                       graph_unique_times: GraphConfig,
                       graph_cum_baseline_hazard: GraphConfig,
                       graph_baseline_survival: GraphConfig,
                       graph_cum_hazard_function: GraphConfig,
                       graph_survival_function: GraphConfig,
                       # graph_confusion_matrix: GraphConfig,
                       # graph_calibration_curve: GraphConfig,
                       # confusion_matrix_parameters: dict = {},
                       # calibration_curve_parameters: dict = {}
                       ):  # ROC curve, precision-recall curve de sklearn
        """
        Creates the different graphs for the breslow estimator.

        Parameters
        ----------
        graph_unique_times : GraphConfig
            A tuple used for the unique times graph.
        graph_cum_baseline_hazard : GraphConfig
            A tuple used for the cumulative baseline hazard graph.
        graph_baseline_survival : GraphConfig
            A tuple used for the baseline survival graph.
        graph_cum_hazard_function : GraphConfig
            A tuple used for the cumulative hazard function graph.
        graph_survival_function : GraphConfig
            A tuple used for the survival function graph.
        graph_confusion_matrix : GraphConfig
            A tuple used for the confusion matrix graph.
        graph_calibration_curve : GraphConfig
            A tuple used for the calibration curve graph.
        confusion_matrix_parameters : dict
            Dictionary of optional parameters for sklearn.metrics.confusion_matrix
        calibration_curve_parameters : dict
            Dictionary of optional parameters for sklearn.calibration.calibration_curve.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            self._breslow_unique_times(TaskInput(task.name, *graph_unique_times))
            self._breslow_cum_baseline_hazard(TaskInput(task.name, *graph_cum_baseline_hazard))
            self._breslow_baseline_survival(TaskInput(task.name, *graph_baseline_survival))
            self._breslow_cum_hazard_function(task, TaskInput(task.name, *graph_cum_hazard_function))
            self._breslow_survival_function(task, TaskInput(task.name, *graph_survival_function))
        # for task in self.tasks.binary_classification_tasks:
        #     self._confusion_matrix(TaskInput(task.name, *graph_confusion_matrix), confusion_matrix_parameters)
        #     self._calibration_curve(TaskInput(task.name, *graph_calibration_curve), calibration_curve_parameters)

    def _breslow_unique_times(self, graph: TaskInput):
        """
        Creates the breslow unique times graph.

        Parameters
        ----------
        graph : TaskInput
            A NamedTuple used for the unique times graph.
        """
        plt.plot(self.breslow_estimator.unique_times_)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_unique_times.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _breslow_cum_baseline_hazard(self, graph: TaskInput):
        """
        Creates the breslow cumulative baseline hazard graph

        Parameters
        ----------
        graph : TaskInput
            A NamedTuple used for the cumulative baseline hazard graph.
        """
        cum_baseline_hazard = self.breslow_estimator.cum_baseline_hazard_
        plt.plot(cum_baseline_hazard.x, cum_baseline_hazard.y)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_cum_baseline_hazard.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _breslow_baseline_survival(self, graph: TaskInput):
        """
        Creates the breslow baseline survival graph.

        Parameters
        ----------
        graph : TaskInput
            A NamedTuple used for the baseline survival graph.
        """
        baseline_survival = self.breslow_estimator.baseline_survival_
        plt.plot(baseline_survival.x, baseline_survival.y)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_baseline_survival.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _breslow_cum_hazard_function(self, task, graph: TaskInput):
        """
        Creates the breslow cumulative hazard function graph.

        Parameters
        ----------
        task
            The task for which to compute the cumulative hazard function.
        graph : TaskInput
            A NamedTuple used for the cumulative hazard function graph.
        """
        prediction = {}
        for prediction_element in self.predictions:
            if prediction.get(task.name, None) is not None:
                prediction[task.name] = torch.cat((prediction.get(task.name), prediction_element[task.name]), dim=-1)
            else:
                prediction[task.name] = (prediction_element[task.name])
        chf_funcs = self.breslow_estimator.get_cumulative_hazard_function(prediction[task.name])
        for fn in chf_funcs:
            plt.step(fn.x, fn(fn.x), where="post")
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_cum_hazard_function.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _breslow_survival_function(self, task, graph: TaskInput):
        """
        Creates the breslow survival function graph.

        Parameters
        ----------
        task
            The task for which to compute the survival function.
        graph : TaskInput
            A NamedTuple used for the survival function graph.
        """
        prediction = {}
        for prediction_element in self.predictions:
            if prediction.get(task.name, None) is not None:
                prediction[task.name] = torch.cat((prediction.get(task.name), prediction_element[task.name]), dim=-1)
            else:
                prediction[task.name] = (prediction_element[task.name])
        survival_funcs = self.breslow_estimator.get_survival_function(prediction[task.name])
        for fn in survival_funcs:
            plt.step(fn.x, fn(fn.x), where="post")
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_survival_function.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _confusion_matrix(self, graph: TaskInput, confusion_matrix_parameters: dict):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : TaskInput
            A NamedTuple used for the confusion matrix graph.
        confusion_matrix_parameters : dict
            Dictionary of optional parameters for sklearn.metrics.confusion_matrix.

        """
        confusion = confusion_matrix(self.ground_truth, self.predictions, **confusion_matrix_parameters)
        plt.imshow(confusion)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_confusion_matrix.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _calibration_curve(self, graph: TaskInput, calibration_curve_parameters: dict):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : TaskInput
            A NamedTuple used for the calibration_curve graph.
        calibration_curve_parameters : dict
            Dictionary of optional parameters for sklearn.calibration.calibration_curve.

        """
        prob_true, prob_pred = calibration_curve(self.ground_truth, self.predictions, **calibration_curve_parameters)
        subplot_x, subplot_y = 2, 1
        fig, arr = plt.subplots(subplot_y, subplot_x)
        arr[0, 0].imshow(prob_pred)
        arr[1, 0].imshow(prob_true)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_calibration_curve.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()
