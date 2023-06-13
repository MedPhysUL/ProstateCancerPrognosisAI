"""
    @file:              dataset_evaluator.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the human user to gauge the
    quality of a model.
"""

from typing import Dict, List, Optional

from src.data.datasets.prostate_cancer import ProstateCancerDataset, TargetsType
from src.models.torch.base.torch_model import Output, TorchModel
from src.metrics.single_task.base import MetricReduction
from src.tools.transforms import to_numpy, batch_to_device
from src.evaluation.single_task.prediction_evaluator import PredictionEvaluator

from monai.data import DataLoader
from torch import cuda, float32, random, tensor
from torch import device as torch_device


class DatasetEvaluator(PredictionEvaluator):
    def __init__(
            self,
            model: TorchModel,
            dataset: ProstateCancerDataset,
            mask: List[int]
    ):
        """Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        model : TorchModel
            The model with which the predictions will be made.
        dataset : ProstateCancerDataset
            The dataset to input to the model.
        mask : List[int]
            Mask determining which patients to use in the dataset.
        """
        self.mask = mask
        self.dataset = dataset
        self.model = model

        ground_truth = []
        subset = self.dataset[self.mask]
        for _, targets in DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None):
            ground_truth.append(targets)

        super().__init__(
            predictions=self._predictions_from_dataset(),
            ground_truth=ground_truth,
            tasks=self.dataset.tasks
        )

    def _predictions_from_dataset(self) -> List[TargetsType]:
        """
        Generates predictions using a dataset, model and mask.

        Returns
        -------
        predictions : List[TargetsType]
            The predictions of the model on the dataset.
        """
        subset = self.dataset[self.mask]
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        predictions = []
        for features, _ in data_loader:
            predictions.append(self.model.predict(features=features))
        return predictions

    @staticmethod
    def score_on_dataset(
            model: TorchModel,
            dataset: ProstateCancerDataset,
            mask: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        model : TorchModel
            The model with which the predictions are to be computed.
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        model.device if model.device else torch_device("cuda") if cuda.is_available() else torch_device("cpu")
        subset = dataset[mask]
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        tasks = dataset.tasks
        table_tasks, seg_tasks = tasks.table_tasks, tasks.segmentation_tasks

        scores = {task.name: {} for task in tasks}
        segmentation_scores = {task.name: {metric.name: [] for metric in task.unique_metrics} for task in seg_tasks}
        table_outputs = {task.name: Output(predictions=[], targets=[]) for task in table_tasks}
        for features, targets in data_loader:
            features, targets = batch_to_device(features, model.device), batch_to_device(targets, model.device)

            predictions = model.predict(features=features)

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
