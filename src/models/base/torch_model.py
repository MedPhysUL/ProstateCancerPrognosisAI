"""
    @file:              torch_model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file contains an abstract class named 'TorchModel' from which all custom pytorch models
                        implemented for the project must inherit. This class allows to store common function of all
                        pytorch models.
"""

from abc import ABC
from typing import Dict, List, NamedTuple, Optional

from monai.data import DataLoader
from numpy import argmin, argmax, linspace
from torch import device as torch_device
from torch import no_grad, round, sigmoid, stack

from ...data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ...metrics.single_task.base import Direction, MetricReduction
from .model import check_if_built, Model
from ...tools.transforms import to_numpy, batch_to_device


def evaluation_function(_func):
    def wrapper(*args, **kwargs):
        self = args[0]

        training = self.training
        self.eval()
        out = _func(*args, **kwargs)
        if training:
            self.train()
        return out

    return wrapper


class Output(NamedTuple):
    predictions: List
    targets: List


class TorchModel(Model, ABC):
    """
    Abstract class used to store common attributes and methods of torch models implemented in the project.
    """

    def __init__(
            self,
            device: Optional[torch_device] = None,
            name: Optional[str] = None
    ) -> None:
        """
        Sets the protected attributes and creates an embedding block if required.

        Parameters
        ----------
        device : Optional[torch_device]
            Device used for training.
        name : Optional[str]
            The name of the model.
        """
        super().__init__(device=device, name=name)

    def build(self, dataset: ProstateCancerDataset):
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        assert all(task.criterion is not None for task in dataset.tasks), (
            f"'TorchModel' requires that all tasks define the 'criterion' attribute at instance initialization."
        )
        super().build(dataset=dataset)
        # table_input_size = len(self._dataset.table_dataset.cont_cols) + len(self._dataset.table_dataset.cat_cols)
        # output size = len(dataset.tasks)

    @check_if_built
    def fix_thresholds_to_optimal_values(
            self
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.
        """
        subset = self._dataset[self._dataset.train_mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)

        tasks, binary_classification_tasks = self._dataset.tasks, self._dataset.tasks.binary_classification_tasks
        outputs_dict = {task.name: Output(predictions=[], targets=[]) for task in binary_classification_tasks}

        thresholds = linspace(start=0.01, stop=0.95, num=95)

        with no_grad():
            for features, targets in data_loader:
                features, targets = batch_to_device(features, self.device), batch_to_device(targets, self.device)

                predictions = self.predict(features, tasks=tasks)

                for task in binary_classification_tasks:
                    outputs_dict[task.name].predictions.append(predictions[task.name].item())
                    outputs_dict[task.name].targets.append(targets[task.name].item())

            for task in binary_classification_tasks:
                output = outputs_dict[task.name]

                for metric in task.metrics:
                    scores = [metric(to_numpy(output.predictions), to_numpy(output.targets), t) for t in thresholds]

                    if metric.direction == Direction.MINIMIZE:
                        metric.threshold = thresholds[argmin(scores)]
                    else:
                        metric.threshold = thresholds[argmax(scores)]

    @check_if_built
    @evaluation_function
    def predict(
            self,
            features: FeaturesType,
            probability: bool = True
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular batch, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.
            - For segmentation tasks, returns the predicted segmentation map.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        predictions = {}
        with no_grad():
            features = batch_to_device(features, self.device)
            outputs = self(features)

            for task in self._tasks.binary_classification_tasks:
                predictions[task.name] = sigmoid(outputs[task.name])
            for task in self._tasks.regression_tasks:
                predictions[task.name] = outputs[task.name]
            for task in self._tasks.segmentation_tasks:
                predictions[task.name] = round(sigmoid(outputs[task.name]))

        return predictions

    @check_if_built
    def predict_on_dataset(
            self,
            mask: List[int],
            probability: bool = True
    ) -> TargetsType:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using the 'mask'
        parameter, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        mask : List[int]
            A list of dataset idx for which we want to obtain the predictions.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions (except segmentation map).
        """
        subset = self._dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)

        predictions = {task.name: [] for task in self._dataset.tasks}
        with no_grad():
            for features, _ in data_loader:
                pred = self.predict(features=features, tasks=self._dataset.tasks, probability=probability)

                for task in self._dataset.tasks.table_tasks:
                    predictions[task.name].append(pred[task.name])

        return {task.name: stack(predictions[task.name], dim=0) for task in self._dataset.tasks.table_tasks}

    @check_if_built
    def score(
            self,
            features: FeaturesType,
            targets: TargetsType
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        targets : TargetsType
            Batch data items.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        with no_grad():
            pred = self.predict(features=features)

            scores = {}
            for task in self._tasks:
                scores[task.name] = {}
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric(to_numpy(pred[task.name]), to_numpy(targets[task.name]))

        return scores

    @check_if_built
    def score_on_dataset(
            self,
            mask: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        subset = self._dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)

        tasks = self._dataset.tasks
        table_tasks, seg_tasks = tasks.table_tasks, tasks.segmentation_tasks

        scores = {task.name: {} for task in tasks}
        segmentation_scores = {task.name: {metric.name: [] for metric in task.unique_metrics} for task in seg_tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}
        with no_grad():
            for features, targets in data_loader:
                features, targets = batch_to_device(features, self.device), batch_to_device(targets, self.device)

                predictions = self.predict(features=features, tasks=tasks)

                for task in seg_tasks:
                    for metric in task.unique_metrics:
                        segmentation_scores[task.name][metric.name].append(
                            metric(predictions[task.name], targets[task.name], MetricReduction.NONE)
                        )

                for task in table_tasks:
                    table_outputs[task.name].predictions.append(predictions[task.name].item())
                    table_outputs[task.name].targets.append(targets[task.name].item())

            for task in seg_tasks:
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric.perform_reduction(
                        segmentation_scores[task.name][metric.name].float()
                    )

            for task in table_tasks:
                output = table_outputs[task.name]
                for metric in task.unique_metrics:
                    scores[task.name][metric.name] = metric(to_numpy(output.predictions), to_numpy(output.targets))

        return scores
