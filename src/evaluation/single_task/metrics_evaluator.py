"""
    @file:              metrics_evaluator.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the human user to gauge the
    quality of a model.
"""
import json
import matplotlib.pyplot as plt
import torch

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from src.data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset
from src.tasks.containers.list import TaskList
from src.models.torch.base.torch_model import Output
from src.metrics.single_task.base import MetricReduction
from src.tools.transforms import to_numpy

from monai.data import DataLoader
from torch import float32, random, tensor, Tensor


class BreslowInput(NamedTuple):
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
        kwargs: Dict[str, Any]
            Dict of positional arguments to pass on to matplotlib.pyplot.savefig()
        """
    task_name: str
    show: bool
    save: Union[bool, str]
    kwargs: Dict[str, Any]


class Evaluator:
    def __init__(
            self,
            model,
            results,
            ground_truth: Optional[List[Union[dict, FeaturesType, Tensor]]] = None,
            mask: Optional[List[int]] = None,
            tasks: Optional[TaskList] = None
            ):
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        model
            Neural network model to evaluate.
        results
            Either the dataset with which the evaluation is desired or the predictions of the model from a dataset.
        ground_truth : Optional[List[Union[dict, FeaturesType, Tensor]]]
            Ground truths to be used as a reference for the computation of the different metrics. This argument is
            required if predictions are used.
        mask : Optional[List[int]]
            Mask of the dataset to specify with which data to use in the evaluation. This argument is required if a
            dataset is used.
        tasks : Optional[Tasklist]
            Object of the class TaskList that specifies for which tasks the model should be evaluated. This argument
            is required if predictions are used.
        """

        self.model = model
        self.ground_truth = ground_truth
        self.mask = mask
        self.tasks = tasks

        if isinstance(results, ProstateCancerDataset):
            self.dataset = results
            self.predictions = self._dataset_to_predictions()
        else:
            self.predictions = results
            self.dataset = None

        if self.dataset is not None:
            assert self.mask is not None, "If a dataset is used, then a mask needs to be provided."
            if self.tasks is None:
                self.tasks = self.dataset.tasks
            if self.ground_truth is None:
                self.ground_truth = []
                subset = self.dataset[self.mask]
                for _, targets in DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None):
                    self.ground_truth.append(targets)

        if self.dataset is None:
            assert self.tasks is not None, "When using predictions, then a list of the tasks is required."
            assert self.ground_truth is not None, "When using predictions, then ground truths are required."

    def _dataset_to_predictions(self):
        """
        Generates predictions using a dataset, model and mask.

        Returns
        -------
        predictions : List[TargetsType]
            The predictions of the model.
        """
        subset = self.dataset[self.mask]
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)
        predictions = []
        for features, _ in data_loader:
            predictions.append(self.model.predict(features=features))
        return predictions

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
                       show: bool = True,
                       save: Union[bool, str] = False
                       ):
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        metrics : Optional[Union[str, List[str]]]
            Either the metric to compute, a list of the metric or "all" which will compute all metrics. Defaults to None
            which computes no metric.
        show : bool
            Whether to print the computed metrics. False means that the metric are computed but not shown. Default to
            True which print the metrics as a dictionary.
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
        elif isinstance(metrics, int):
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

        if show:
            print(scalar_metrics)
        if save:
            path = save + '/scalar_metrics.json'
            with open(path, 'w') as file_path:
                json.dump(scalar_metrics, file_path)

    def breslow_graphs(self,
                       graph_unique_times: Tuple[bool, Union[bool, str], Dict[str, Any]],
                       graph_cum_baseline_hazard: Tuple[bool, Union[bool, str], Dict[str, Any]],
                       graph_baseline_survival: Tuple[bool, Union[bool, str], Dict[str, Any]],
                       graph_cum_hazard_function: Tuple[bool, Union[bool, str], Dict[str, Any]],
                       graph_survival_function: Tuple[bool, Union[bool, str], Dict[str, Any]]
                       ):
        """
        Creates the different graphs for the breslow estimator.

        Parameters
        ----------
        graph_unique_times: Tuple[bool, Union[bool, str], Dict[str, Any]]
            A tuple used for the unique times graph with: a bool to decide whether to show the graph, either the path
            to the folder where the graph should be saved of False, kwargs for matplotlib.pyplot.savefig().
        graph_cum_baseline_hazard: Tuple[bool, Union[bool, str], Dict[str, Any]]
            A tuple used for the cumulative baseline hazard graph with: a bool to decide whether to show the graph,
            either the path to the folder where the graph should be saved of False, kwargs for
            matplotlib.pyplot.savefig().
        graph_baseline_survival: Tuple[bool, Union[bool, str], Dict[str, Any]]
            A tuple used for the baseline survival graph with: a bool to decide whether to show the graph, either the
            path to the folder where the graph should be saved of False, kwargs for matplotlib.pyplot.savefig().
        graph_cum_hazard_function: Tuple[bool, Union[bool, str], Dict[str, Any]]
            A tuple used for the cumulative hazard function graph with: a bool to decide whether to show the graph,
            either the path to the folder where the graph should be saved of False, kwargs for
            matplotlib.pyplot.savefig().
        graph_survival_function: Tuple[bool, Union[bool, str], Dict[str, Any]]
            A tuple used for the survival function graph with: a bool to decide whether to show the graph, either the
            path to the folder where the graph should be saved of False, kwargs for matplotlib.pyplot.savefig().
        """
        for task in self.tasks.survival_analysis_tasks:
            self.breslow_estimator = task.breslow_estimator
            self._breslow_unique_times(BreslowInput(task.name, *graph_unique_times))
            self._breslow_cum_baseline_hazard(BreslowInput(task.name, *graph_cum_baseline_hazard))
            self._breslow_baseline_survival(BreslowInput(task.name, *graph_baseline_survival))
            self._breslow_cum_hazard_function(task, BreslowInput(task.name, *graph_cum_hazard_function))
            self._breslow_survival_function(task, BreslowInput(task.name, *graph_survival_function))

    def _breslow_unique_times(self, graph: BreslowInput):
        """
        Creates the breslow unique times graph.

        Parameters
        ----------
        graph: BreslowInput
            A NamedTuple used for the unique times graph with: the name of the task, a bool to decide whether to show
            the graph, either the path to the folder where the graph should be saved of False, kwargs for
            matplotlib.pyplot.savefig().
        """
        plt.plot(self.breslow_estimator.unique_times_)
        if graph.save:
            path = graph.save + f'/{graph.task_name}_breslow_unique_times.pdf'
            plt.savefig(path, **graph.kwargs)
        if graph.show:
            plt.show()
        if not graph.show:
            plt.close()

    def _breslow_cum_baseline_hazard(self, graph: BreslowInput):
        """
        Creates the breslow cumulative baseline hazard graph

        Parameters
        ----------
        graph: BreslowInput
            A NamedTuple used for the cumulative baseline hazard graph with: the name of the task, a bool to decide
            whether to show the graph, either the path to the folder where the graph should be saved of False, kwargs
            for matplotlib.pyplot.savefig().
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

    def _breslow_baseline_survival(self, graph: BreslowInput):
        """
        Creates the breslow baseline survival graph.

        Parameters
        ----------
        graph: BreslowInput
            A NamedTuple used for the baseline survival graph with: the name of the task, a bool to decide whether to
            show the graph, either the path to the folder where the graph should be saved of False, kwargs for
            matplotlib.pyplot.savefig().
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

    def _breslow_cum_hazard_function(self, task, graph: BreslowInput):
        """
        Creates the breslow cumulative hazard function graph.

        Parameters
        ----------
        task
            The task for which to compute the cumulative hazard function.
        graph: BreslowInput
            A NamedTuple used for the cumulative hazard function graph with: the name of the task, a bool to decide
            whether to show the graph, either the path to the folder where the graph should be saved of False, kwargs
            for matplotlib.pyplot.savefig().
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

    def _breslow_survival_function(self, task, graph: BreslowInput):
        """
        Creates the breslow survival function graph.

        Parameters
        ----------
        task
            The task for which to compute the survival function.
        graph: BreslowInput
            A NamedTuple used for the survival function graph with: the name of the task, a bool to decide whether to
            show the graph, either the path to the folder where the graph should be saved of False, kwargs for
            matplotlib.pyplot.savefig().
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
