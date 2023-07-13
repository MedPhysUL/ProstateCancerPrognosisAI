"""
    @file:              torch_model.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 04/2023

    @Description:       This file contains an abstract class named 'TorchModel' from which all custom pytorch models
                        implemented for the project must inherit. This class allows to store common function of all
                        pytorch models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from monai.data import DataLoader
from torch import device as torch_device
from torch import no_grad, random, round, sigmoid, stack

from ...base import check_if_built, Model
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ....evaluation.model_evaluator import ModelEvaluator
from ....tools.transforms import to_numpy, batch_to_device


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


class TorchModel(Model, ABC):
    """
    Abstract class used to store common attributes and methods of torch models implemented in the project.
    """

    def __init__(
            self,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
            bayesian: bool = False
    ) -> None:
        """
        Sets the protected attributes and creates an embedding block if required.

        Parameters
        ----------
        device : Optional[torch_device]
            Device used for training.
        name : Optional[str]
            The name of the model.
        seed : Optional[int]
            Random state used for reproducibility.
        """
        super().__init__(device=device, name=name, seed=seed)

        self._bayesian = bayesian

    @property
    def bayesian(self) -> bool:
        """
        Returns bayesian status.

        Returns
        -------
        bayesian_status : bool
            Whether the model is in bayesian mode.
        """
        return self._bayesian

    def build(self, dataset: ProstateCancerDataset) -> TorchModel:
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : TorchModel
            The current model.
        """
        assert all(task.criterion is not None for task in dataset.tasks), (
            f"'TorchModel' requires that all tasks define the 'criterion' attribute at instance initialization."
        )
        super().build(dataset=dataset)
        return self

    @check_if_built
    @abstractmethod
    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Executes the forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        raise NotImplementedError

    @check_if_built
    @evaluation_function
    @no_grad()
    def fit_breslow_estimators(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fit all survival analysis tasks' breslow estimators given the training dataset. It is recommended to train the
        model before calling this method.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        super().fit_breslow_estimators(dataset)

    @check_if_built
    @evaluation_function
    @no_grad()
    def fix_thresholds_to_optimal_values(
            self,
            dataset: ProstateCancerDataset
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        ModelEvaluator.fix_thresholds_to_optimal_values_with_dataset(model=self, dataset=dataset)

    @check_if_built
    @evaluation_function
    @no_grad()
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
            - For survival analysis tasks, returns the estimated risk of experiencing an event.
            - For segmentation tasks, returns the predicted segmentation map.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression, survival and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        predictions = {}
        features = batch_to_device(features, self.device)
        outputs = self(features)

        for task in self._tasks.binary_classification_tasks:
            if probability:
                predictions[task.name] = sigmoid(outputs[task.name])
            else:
                predictions[task.name] = (sigmoid(outputs[task.name]) >= task.decision_threshold_metric.threshold)
        for task in self._tasks.regression_tasks:
            predictions[task.name] = outputs[task.name]
        for task in self._tasks.survival_analysis_tasks:
            predictions[task.name] = outputs[task.name]
        for task in self._tasks.segmentation_tasks:
            predictions[task.name] = round(sigmoid(outputs[task.name]))

        return predictions

    @check_if_built
    @no_grad()
    def predict_on_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: Optional[List[int]] = None,
            probability: bool = True
    ) -> Optional[TargetsType]:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using the 'mask'
        parameter, particularly :
            - For binary classification tasks, returns the probability of belonging to class 1 OR directly returns the
              predicted class, depending on the value of the 'probability' parameter.
            - For regression tasks, returns the predicted real-valued target.
            - For survival analysis tasks, returns the estimated risk of experiencing an event.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : Optional[List[int]]
            A list of dataset idx for which we want to obtain the predictions. If no mask is given, all patients are
            used.
        probability : bool
            Whether to return probability predictions or class predictions for binary classification task predictions.
            Doesn't affect regression, survival and segmentation tasks predictions.

        Returns
        -------
        predictions : TargetsType
            Predictions (except segmentation map).
        """
        subset = dataset[mask] if mask is not None else dataset
        rng_state = random.get_rng_state()
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        random.set_rng_state(rng_state)

        predictions = {task.name: [] for task in dataset.tasks}
        for features, _ in data_loader:
            pred = self.predict(features=features, probability=probability)

            for task in dataset.tasks.table_tasks:
                predictions[task.name].append(pred[task.name])

        if dataset.tasks.table_tasks:
            return {task.name: stack(predictions[task.name], dim=0) for task in dataset.tasks.table_tasks}
        else:
            return None

    @check_if_built
    @no_grad()
    def compute_score(
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
            Score for each task and each metric.
        """
        pred = self.predict(features=features)

        scores = {}
        for task in self._tasks:
            scores[task.name] = {}
            for metric in task.unique_metrics:
                scores[task.name][metric.name] = metric(to_numpy(pred[task.name]), to_numpy(targets[task.name]))

        return scores

    @check_if_built
    @no_grad()
    def compute_score_on_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each task and each metric.
        """
        return ModelEvaluator.compute_score_on_dataset(model=self, dataset=dataset, mask=mask)
