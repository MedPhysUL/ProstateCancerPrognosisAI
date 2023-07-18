"""
    @file:              model_evaluator.py
    @Author:            Felix Desroches, Maxence Larose

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the user to gauge the
    quality of a model.
"""

import json
from typing import Dict, List, Optional, Union

from monai.data import DataLoader
import numpy as np
from torch import float32, random, tensor

from ..data.datasets.prostate_cancer import ProstateCancerDataset
from ..evaluation.prediction_evaluator import PredictionEvaluator, Output
from ..metrics.single_task.base import MetricReduction
from ..models.base.model import Model
from ..tools.transforms import batch_to_device, to_numpy


class ModelEvaluator(PredictionEvaluator):
    """
    Class used to show metrics and graphs for the user to gauge the quality of a model. Inherits from the class used to
    evaluate predictions. Can compute metrics for segmentation tasks and table tasks.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
            n_samples: int = 10
    ) -> None:
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        model : Model
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        n_samples : int
            Number of samples to use for bayesian inference. Only used if the model is in bayesian mode. Defaults to 10.
        """
        self.dataset = dataset
        self.model = model

        super().__init__(
            predictions=self.model.predict_on_dataset(dataset=self.dataset, n_samples=n_samples),
            targets=self.dataset.table_dataset.y,
            tasks=self.dataset.tasks,
            breslow_mask=self.dataset.train_mask,
            fit_breslow_estimators=False
        )

    @staticmethod
    def compute_score_on_dataset(
            model: Model,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None,
            n_samples: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using the mask parameter
        given when instantiating the ModelEvaluator object. Can compute metrics for segmentation tasks.

        Parameters
        ----------
        model : Model
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        mask : Optional[List[int]]
            Mask to use when selecting the patients with which the metrics are computed. If no mask is given, all
            patients are used.
        n_samples : int
            Number of samples to use for bayesian inference. Only used if the model is in bayesian mode. Defaults to 10.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        device = model.device
        subset = dataset[mask] if mask is not None else dataset
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        tasks = dataset.tasks
        table_tasks, seg_tasks = tasks.table_tasks, tasks.segmentation_tasks

        scores = {task.name: {} for task in tasks}
        segmentation_scores = {task.name: {metric.name: [] for metric in task.unique_metrics} for task in seg_tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}
        for features, targets in data_loader:
            features, targets = batch_to_device(features, device), batch_to_device(targets, device)

            predictions = model.predict(features=features, n_samples=n_samples)

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

    @staticmethod
    def fix_thresholds_to_optimal_values_with_dataset(
            model: Model,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None,
            n_samples: int = 10
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        model : Model
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        mask : Optional[List[int]]
            Mask used to specify which patients to use when optimizing the thresholds. Uses train mask by default.
        n_samples : int
            Number of samples to use for bayesian inference. Only used if the model is in bayesian mode. Defaults to 10.
        """
        if mask is None:
            mask = dataset.train_mask

        subset = dataset[mask]
        device = model.device
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        binary_classification_tasks = dataset.tasks.binary_classification_tasks
        outputs_dict = {task.name: Output(predictions=[], targets=[]) for task in binary_classification_tasks}
        thresholds = np.linspace(start=0.01, stop=0.95, num=95)

        for features, targets in data_loader:
            features, targets = batch_to_device(features, device), batch_to_device(targets, device)

            predictions = model.predict(features=features, n_samples=n_samples)

            for task in binary_classification_tasks:
                outputs_dict[task.name].predictions.append(predictions[task.name].item())
                outputs_dict[task.name].targets.append(targets[task.name].item())

        PredictionEvaluator._fix_metric_threshold(
            binary_classification_tasks=binary_classification_tasks,
            outputs_dict=outputs_dict,
            thresholds=thresholds
        )

    def compute_score(
            self,
            mask: Optional[List[int]] = None,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        mask : Optional[List[int]]
            Mask used to specify with which patients to compute the metrics. Defaults to the mask given when creating
            the ModelEvaluator object.
        path_to_save_folder : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Dictionary of the metrics of each applicable task.
        """
        scores = self.compute_score_on_dataset(model=self.model, dataset=self.dataset, mask=mask)

        if path_to_save_folder is not None:
            with open(f"{path_to_save_folder}/metrics.json", "w") as file_path:
                json.dump(scores, file_path)
        return scores

    def plot_confusion_matrix(
            self,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            threshold: Optional[Union[int, List[int], slice]] = None,
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
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.confusion_matrix.
        """
        if not isinstance(threshold, int):
            self.fix_thresholds_to_optimal_values_with_dataset(model=self.model, dataset=self.dataset, mask=threshold)

        self._create_confusion_matrix_from_thresholds(
            show=show,
            path_to_save_folder=path_to_save_folder,
            threshold=threshold,
            **kwargs
        )
