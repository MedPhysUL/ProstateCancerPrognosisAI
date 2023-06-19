"""
    @file:              model_evaluator.py
    @Author:            Felix Desroches

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

from ...data.datasets.prostate_cancer import ProstateCancerDataset
from ...evaluation.single_task.prediction_evaluator import PredictionEvaluator, Output
from ...metrics.single_task.base import MetricReduction
from ...models.base.model import Model
from ...tools.transforms import batch_to_device, to_numpy


class ModelEvaluator(PredictionEvaluator):
    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None
    ) -> None:
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        model : Model
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        mask : Optional[List[int]]
            Mask determining which patients to use in the dataset to build the predictions and targets lists. Defaults
            to dataset.test_mask.
        """
        if mask is None:
            self.mask = dataset.test_mask
        else:
            self.mask = mask
        self.dataset = dataset
        self.model = model

        super().__init__(
            predictions=self.model.predict_on_dataset(dataset=self.dataset, mask=self.mask),
            targets=self.dataset.table_dataset[self.mask].y,
            tasks=self.dataset.tasks
        )

    def _compute_dataset_score(
            self,
            mask: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using the mask parameter
        given when instantiating the ModelEvaluator object. Can compute metrics for segmentation tasks.

        Parameters
        ----------
        mask : List[int]
            Mask to use when selecting the patients with which the metrics are computed.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        device = self.model.device
        subset = self.dataset[self.mask]
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        tasks = self.dataset.tasks
        table_tasks, seg_tasks = tasks.table_tasks, tasks.segmentation_tasks

        scores = {task.name: {} for task in tasks}
        segmentation_scores = {task.name: {metric.name: [] for metric in task.unique_metrics} for task in seg_tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}
        for features, targets in data_loader:
            features, targets = batch_to_device(features, device), batch_to_device(targets, device)

            predictions = self.model.predict(features=features)

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

    def fix_thresholds_to_optimal_values(
            self,
            mask: Optional[List[int]] = None
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        mask : Optional[List[int]]
            Mask used to specify which patients to use when optimizing the thresholds. Uses train mask by default.
        """
        if mask is None:
            mask = self.dataset.train_mask
        subset = self.dataset[mask]
        device = self.model.device
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        binary_classification_tasks = self.dataset.tasks.binary_classification_tasks
        outputs_dict = {task.name: Output(predictions=[], targets=[]) for task in binary_classification_tasks}
        thresholds = np.linspace(start=0.01, stop=0.95, num=95)

        for features, targets in data_loader:
            features, targets = batch_to_device(features, device), batch_to_device(targets, device)

            predictions = self.model.predict(features)

            for task in binary_classification_tasks:
                outputs_dict[task.name].predictions.append(predictions[task.name].item())
                outputs_dict[task.name].targets.append(targets[task.name].item())

        self._fix_metric_threshold(
            binary_classification_tasks=binary_classification_tasks,
            outputs_dict=outputs_dict,
            thresholds=thresholds
        )

    def compute_metrics(
            self,
            mask: Optional[List[int]] = None,
            save_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        mask : Optional[List[int]]
            Mask used to specify with which patients to compute the metrics. Defaults to the mask given when creating
            the ModelEvaluator object.
        save_path : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Dictionary of the metrics of each applicable task.
        """
        if mask is None:
            mask = self.mask
        scores = self._compute_dataset_score(mask)

        if save_path is not None:
            with open(f"{save_path}/metrics.json", "w") as file_path:
                json.dump(scores, file_path)
        return scores
