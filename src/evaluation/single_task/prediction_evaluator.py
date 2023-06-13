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

from src.data.datasets.prostate_cancer import FeaturesType, TargetsType
from src.tasks.containers.list import TaskList
from src.models.torch.base.torch_model import Output
from src.metrics.single_task.base import MetricReduction
from src.tools.transforms import to_numpy

from torch import float32, tensor, Tensor
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


class GraphConfig(NamedTuple):
    """
    Inputs for metric graphs.

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


class PredictionEvaluator:
    def __init__(
            self,
            predictions: List[TargetsType],
            ground_truth: List[Union[dict, FeaturesType, Tensor]],
            tasks: TaskList
    ):
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        predictions : List[TargetsType]
            The predictions of the model from a dataset.
        ground_truth : List[Union[dict, FeaturesType, Tensor]]
            Ground truths to be used as a reference for the computation of the different metrics.
        tasks : Tasklist
            Object of the class TaskList that specifies for which tasks the model should be evaluated.
        """

        self.breslow_estimator = None
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

    def scalar_metrics(
            self,
            metrics: Optional[Union[str, List[str]]] = None,
            return_metrics: bool = True,
            save: Optional[str] = None
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
        scores = self._score_on_predictions()

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

        for task, metric_dict in scores.items():
            scalar_metrics[task] = {}
            metric_present = False
            for metric, value in metric_dict.items():
                if metric in metrics:
                    metric_present = True
                    scalar_metrics[task][metric] = value
            if not metric_present:
                scalar_metrics[task] = "N/A"

        if save is not None:
            path = save + '/scalar_metrics.json'
            with open(path, 'w') as file_path:
                json.dump(scalar_metrics, file_path)
        if return_metrics:
            return scalar_metrics

    def visual_metrics(
            self,
            graph_configurations: GraphConfig,
            threshold: float = 0.5
    ):
        """
        Creates the different graphs for metrics that can be visualised in a 2D graph.

        Parameters
        ----------
        graph_configurations : GraphConfig
            NamedTuple to use with each graph.
        threshold : float
            Threshold to use for the confusion matrix when going from continuous probability to binary classification.
        """

        self.graph_breslow_unique_times(graph_configurations)
        self.graph_breslow_cum_baseline_hazard(graph_configurations)
        self.graph_breslow_baseline_survival(graph_configurations)
        self.graph_breslow_cum_hazard_function(graph_configurations)
        self.graph_breslow_survival_function(graph_configurations)
        self.graph_confusion_matrix(graph_configurations, threshold, {})
        self.graph_calibration_curve(graph_configurations, {})
        self.graph_roc_curve(graph_configurations, {})
        self.graph_precision_recall_curve(graph_configurations, {})

    def graph_breslow_unique_times(
            self,
            graph: GraphConfig
    ):
        """
        Creates the breslow unique times graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the unique times graph.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            plt.plot(self.breslow_estimator.unique_times_)

            if graph.save:
                path = graph.save + f'/{task.name}_breslow_unique_times.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_breslow_cum_baseline_hazard(
            self,
            graph: GraphConfig
    ):
        """
        Creates the breslow cumulative baseline hazard graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the cumulative baseline hazard graph.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            cum_baseline_hazard = self.breslow_estimator.cum_baseline_hazard_
            plt.plot(cum_baseline_hazard.x, cum_baseline_hazard.y)

            if graph.save:
                path = graph.save + f'/{task.name}_breslow_cum_baseline_hazard.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_breslow_baseline_survival(
            self,
            graph: GraphConfig
    ):
        """
        Creates the breslow baseline survival graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the baseline survival graph.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            baseline_survival = self.breslow_estimator.baseline_survival_
            plt.plot(baseline_survival.x, baseline_survival.y)

            if graph.save:
                path = graph.save + f'/{task.name}_breslow_baseline_survival.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_breslow_cum_hazard_function(
            self,
            graph: GraphConfig
    ):
        """
        Creates the breslow cumulative hazard function graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the cumulative hazard function graph.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            prediction = {}
            for prediction_element in self.predictions:
                if prediction.get(task.name, False):
                    prediction[task.name] = torch.cat(
                        (prediction.get(task.name), prediction_element[task.name]),
                        dim=-1
                    )
                else:
                    prediction[task.name] = (prediction_element[task.name])
            for chf_func in self.breslow_estimator.get_cumulative_hazard_function(prediction[task.name]):
                plt.step(chf_func.x, chf_func(chf_func.x), where="post")

            if graph.save:
                path = graph.save + f'/{task.name}_breslow_cum_hazard_function.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_breslow_survival_function(
            self,
            graph: GraphConfig
    ):
        """
        Creates the breslow survival function graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the survival function graph.
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            prediction = {}
            for prediction_element in self.predictions:
                if prediction.get(task.name, False):
                    prediction[task.name] = torch.cat(
                        (prediction.get(task.name), prediction_element[task.name]),
                        dim=-1
                    )
                else:
                    prediction[task.name] = (prediction_element[task.name])
            for survival_func in self.breslow_estimator.get_survival_function(prediction[task.name]):
                plt.step(survival_func.x, survival_func(survival_func.x), where="post")

            if graph.save:
                path = graph.save + f'/{task.name}_breslow_survival_function.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_confusion_matrix(
            self,
            graph: GraphConfig,
            threshold: float,
            confusion_matrix_parameters: dict
    ):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the confusion matrix graph.
        threshold : float
            Threshold to consider while transforming continuous predictions to binary results
        confusion_matrix_parameters : dict
            Dictionary of optional parameters for sklearn.metrics.confusion_matrix.

        """
        for task in self.tasks.binary_classification_tasks:
            y_true, y_pred = [], []
            for ground_truth in self.ground_truth:
                y_true.append(ground_truth[task.name][0])
            for predictions in self.predictions:
                if predictions[task.name][0] > threshold:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            plt.imshow(confusion_matrix(y_true, y_pred, **confusion_matrix_parameters))

            if graph.save:
                path = graph.save + f'/{task.name}_confusion_matrix.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_calibration_curve(
            self,
            graph: GraphConfig,
            calibration_curve_parameters: dict
    ):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the calibration_curve graph.
        calibration_curve_parameters : dict
            Dictionary of optional parameters for sklearn.calibration.calibration_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            y_true, y_pred = [], []
            for ground_truth in self.ground_truth:
                y_true.append(ground_truth[task.name][0])
            for predictions in self.predictions:
                y_pred.append(predictions[task.name][0])
            prob_true, prob_pred = calibration_curve(y_true, y_pred, **calibration_curve_parameters)
            plt.plot(prob_true, prob_pred, 'go')
            plt.plot([1, 0], [1, 0], 'k')

            if graph.save:
                path = graph.save + f'/{task.name}_calibration_curve.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_roc_curve(
            self,
            graph: GraphConfig,
            roc_curve_parameters: dict
    ):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the roc_curve graph.
        roc_curve_parameters : dict
            Dictionary of optional parameters for sklearn.metrics.roc_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            y_true, y_pred = [], []
            for ground_truth in self.ground_truth:
                y_true.append(ground_truth[task.name][0])
            for predictions in self.predictions:
                y_pred.append(predictions[task.name][0])
            fpr, tpr, threshold = roc_curve(y_true, y_pred, **roc_curve_parameters)
            plt.plot(fpr, tpr, 'g')
            plt.plot([1, 0], [1, 0], 'k')

            if graph.save:
                path = graph.save + f'/{task.name}_roc_curve.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()

    def graph_precision_recall_curve(
            self,
            graph: GraphConfig,
            precision_recall_curve_parameters: dict
    ):
        """
        Creates the confusion matrix graph.

        Parameters
        ----------
        graph : GraphConfig
            A NamedTuple used to configure the precision_recall_curve graph.
        precision_recall_curve_parameters : dict
            Dictionary of optional parameters for sklearn.metrics.precision_recall_curve.

        """
        for task in self.tasks.binary_classification_tasks:
            y_true, y_pred = [], []
            for ground_truth in self.ground_truth:
                y_true.append(ground_truth[task.name][0])
            for predictions in self.predictions:
                y_pred.append(predictions[task.name][0])
            precision, recall, threshold = precision_recall_curve(y_true, y_pred, **precision_recall_curve_parameters)
            plt.step(recall, precision, 'g')

            if graph.save:
                path = graph.save + f'/{task.name}_precision_recall_curve.pdf'
                plt.savefig(path, **graph.kwargs)
            if graph.show:
                plt.show()
            else:
                plt.close()
