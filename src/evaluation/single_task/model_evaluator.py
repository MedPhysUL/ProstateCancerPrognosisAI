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

import matplotlib.pyplot as plt
import numpy as np
from monai.data import DataLoader
from sklearn.metrics import confusion_matrix
import torch
from torch import cuda, float32, random, tensor
from torch import device as torch_device

from ...data.datasets.prostate_cancer import ProstateCancerDataset, TargetsType
from ...evaluation.single_task.prediction_evaluator import PredictionEvaluator
from ...metrics.single_task.base import MetricReduction
from ...models.base.model import Model
from ...models.torch.base.torch_model import Output
from ...tools.transforms import batch_to_device, to_numpy


class ModelEvaluator(PredictionEvaluator):
    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None
    ):
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        model : Model
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        mask : Optional[List[int]]
            Mask determining which patients to use in the dataset to build the predictions and targets lists.
        """
        if mask is None:
            self.mask = dataset.test_mask
        else:
            self.mask = mask
        self.dataset = dataset
        self.model = model

        super().__init__(
            predictions=self._predictions_from_dataset(),
            ground_truth=self._targets_from_dataset(),
            tasks=self.dataset.tasks
        )

    def _predictions_from_dataset(self) -> List[TargetsType]:
        """
        Generates predictions using a dataset, model and mask.

        Returns
        -------
        predictions : List[TargetsType]
            The predictions of the model on the dataset in a list.
        """
        feature_dict = self.model.predict_on_dataset(dataset=self.dataset, mask=self.mask)
        dataset_length = len(feature_dict[list(feature_dict.keys())[0]])
        feature_list = [{} for _ in range(dataset_length)]
        for task in self.dataset.tasks.table_tasks:
            features = feature_dict.get(task.name).tolist()
            for i, feature in enumerate(features):
                feature_list[i][task.name] = tensor(feature)

        return feature_list

    def _targets_from_dataset(self) -> List[TargetsType]:
        """
        Returns
        ------
        targets : List[TargetsType]
            The targets in a dataset in a list.
        """
        targets_dict = self.dataset.table_dataset[self.mask].y
        dataset_length = len(targets_dict[list(targets_dict.keys())[0]])
        targets_list = [{} for _ in range(dataset_length)]
        for task in self.dataset.tasks.table_tasks:
            targets = targets_dict.get(task.name).tolist()
            for i, target in enumerate(targets):
                if isinstance(target, int):
                    targets_list[i][task.name] = tensor([target])
                else:
                    targets_list[i][task.name] = tensor([target], dtype=torch.float64)

        return targets_list

    def _compute_dataset_score(self) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using the mask parameter
        given when instantiating the ModelEvaluator object. Can compute metrics for segmentation tasks.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        self.model.device if self.model.device else torch_device("cuda") if cuda.is_available() else torch_device("cpu")
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

    def compute_dataset_metrics(
            self,
            save_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes the metrics associated with each task.

        Parameters
        ----------
        save_path : Union[bool, str]
            Whether to save the computed metrics. If saving the metrics is desired, then this is the path of the folder
            where they will be saved as a json file. Defaults to False which does not save the metrics.
        """
        scores = self._compute_dataset_score()

        if save_path is not None:
            with open(f'{save_path}/scalar_metrics.json', 'w') as file_path:
                json.dump(scores, file_path)
        return scores

    def plot_dataset_confusion_matrix(
            self,
            show: bool,
            save: Optional[str] = None,
            **kwargs
    ):
        """
        Creates the confusion matrix graph. Takes the mask parameter given when instantiating the ModelEvaluator object
        to select the patients used to optimize the threshold used when computing binary classification from continuous
        probability.

        Parameters
        ----------
        show : bool
            Whether to show the graph.
        save : Optional[str],
            Whether to save the graph, if so, then this value is the path to the save folder.
        kwargs
            These arguments will be passed on to matplotlib.pyplot.savefig and sklearn.metrics.confusion_matrix.

        """
        for task in self.tasks.binary_classification_tasks:
            self.model.fix_thresholds_to_optimal_values()
            threshold = task.decision_threshold_metric.threshold
            y_true, y_pred = [], []
            for ground_truth in self.targets:
                y_true.append(ground_truth[task.name][0])
            for predictions in self.predictions:
                y_pred += [1] if predictions[task.name][0] >= threshold else [0]

            plt.imshow(confusion_matrix(
                y_true,
                y_pred,
                labels=kwargs.get('labels', None),
                sample_weight=kwargs.get('sample_weight', None),
                normalize=kwargs.get('normalize', None)
            ))

            if save is not None:
                path = save + f'/{task.name}_confusion_matrix.pdf'
                plt.savefig(path, **kwargs)
            if show:
                plt.show()
            else:
                plt.close()
