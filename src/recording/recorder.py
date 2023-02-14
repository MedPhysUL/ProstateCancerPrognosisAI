"""
    @file:              recorder.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:      This file is used to define the Recorder class.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional

from torch import Tensor, save, zeros
from torch.nn import Module

from src.data.processing.tools import MaskType
from src.models.base.model import BaseModel
from src.recording.constants import *


class Recorder:
    """
    Recorder objects used save results of the experiments
    """

    # Dictionary that associate the mask types to their proper section
    MASK_TO_SECTION = {
        METRICS: {
            MaskType.TRAIN: TRAIN_METRICS,
            MaskType.TEST: TEST_METRICS,
            MaskType.VALID: VALID_METRICS
        },
        RESULTS: {
            MaskType.TRAIN: TRAIN_RESULTS,
            MaskType.TEST: TEST_RESULTS,
            MaskType.VALID: VALID_RESULTS}
    }

    def __init__(
            self,
            evaluation_name: str,
            index: int,
            recordings_path: str
    ):
        """
        Sets protected attributes.

        Parameters
        ----------
        evaluation_name : str
            Name of the evaluation.
        index : int
            Index of the outer split.
        recordings_path : str
            Path leading to where we want to save the results.
        """

        # We store the protected attributes
        self._data = {
            NAME: evaluation_name,
            INDEX: index,
            DATA_INFO: {},
            HYPERPARAMETERS: {},
            HYPERPARAMETER_IMPORTANCE: {},
            FEATURE_IMPORTANCE: {},
            TRAIN_METRICS: {},
            TEST_METRICS: {},
            VALID_METRICS: {},
            COEFFICIENT: {},
            TRAIN_RESULTS: {},
            TEST_RESULTS: {},
            VALID_RESULTS: {}
        }

        self._path = os.path.join(recordings_path, evaluation_name, f"Split_{index}")

        # We create the folder where the information will be saved
        os.makedirs(self._path, exist_ok=True)

    def generate_file(self) -> None:
        """
        Save the protected dictionary into a json file.
        """
        # We remove empty sections
        self._data = {k: v for k, v in self._data.items() if (k in [NAME, INDEX] or len(v) != 0)}

        # We save all the data collected in a json file
        filepath = os.path.join(self._path, RECORDS_FILE)
        with open(filepath, "w") as file:
            json.dump(self._data, file, indent=True)

    def record_coefficient(
            self,
            name: str,
            value: float
    ) -> None:
        """
        Saves the value associated to a coefficient (used for linear regression).

        Parameters
        ----------
        name : str
            Name of the variable associated to the coefficient.
        value : float
            Value of the coefficient.
        """
        self._data[COEFFICIENT][name] = value

    def record_data_info(
            self,
            data_name: str,
            data: Any
    ) -> None:
        """
        Records the specific value "data" associated to the variable "data_name" in the protected dictionary.

        Parameters
        ----------
        data_name : str
            Name of the variable for which we want to save a specific information.
        data : Any
            Value we want to store.
        """
        self._data[DATA_INFO][data_name] = data

    def record_features_importance(
            self,
            feature_importance: Dict[str, float]
    ) -> None:
        """
        Saves the features' importance in the protected dictionary.

        Parameters
        ----------
        feature_importance : str
            Dictionary of features and their importance.
        """
        # We save all the hyperparameters importance
        for key in feature_importance.keys():
            self._data[FEATURE_IMPORTANCE][key] = round(feature_importance[key], 4)

    def record_hyperparameters(
            self,
            hyperparameters: Dict[str, Any]
    ) -> None:
        """
        Saves the hyperparameters in the protected dictionary.

        Parameters
        ----------
        hyperparameters : Dict[str, Any]
            Dictionary of hyperparameters and their value.
        """
        # We save all the hyperparameters
        for key in hyperparameters.keys():
            self._data[HYPERPARAMETERS][key] = round(hyperparameters[key], 6) if \
                isinstance(hyperparameters[key], float) else hyperparameters[key]

    def record_hyperparameters_importance(
            self,
            hyperparameter_importance: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Saves the hyperparameters' importance in the protected dictionary.

        Parameters
        ----------
        hyperparameter_importance : Dict[str, Dict[str, float]]
            Dictionary of hyperparameters and their importance for each tasks.
        """
        for task_name, hps_importance in hyperparameter_importance.items():
            self._data[HYPERPARAMETER_IMPORTANCE][task_name] = {}

            # We save all the hyperparameters importance
            for key in hps_importance.keys():
                self._data[HYPERPARAMETER_IMPORTANCE][task_name][key] = round(hps_importance[key], 4)

    def record_model(
            self,
            model: BaseModel
    ) -> None:
        """
        Saves a model using pickle or torch's save function.

        Parameters
        ----------
        model : BaseModel
            Model to save
        """
        # If the model is a torch module with save it using torch
        if isinstance(model, Module):
            save(model, os.path.join(self._path, "model.pt"))
        else:
            # We save the model with pickle
            filepath = os.path.join(self._path, "model.sav")
            pickle.dump(model, open(filepath, "wb"))

    def record_scores(
            self,
            score: float,
            task: str,
            metric: str,
            mask_type: str = MaskType.TRAIN
    ) -> None:
        """
        Saves the score associated to a metric.

        Parameters
        ----------
        score : float
            Metric score.
        task : str
            Name of the task.
        metric : str
            Name of the metric.
        mask_type : str
            Train, test or valid.
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[METRICS][mask_type]

        if not (task in self._data[section]):
            self._data[section][task] = {}

        # We save the score of the given metric
        self._data[section][task][metric] = round(score, 6)

    def record_predictions(
            self,
            ids: List[str],
            predictions: Tensor,
            targets: Optional[Tensor],
            mask_type: str = MaskType.TRAIN
    ) -> None:
        """
        Save the predictions of a given model for each patient ids.

        Parameters
        ----------
        ids : List[str]
            Patient/participant ids
        predictions : Tensor
            Predicted class or regression value.
        targets : Optional[Tensor]
            Target value.
        mask_type : str
            Train, test or valid.
        """
        # We find the proper section name
        section = Recorder.MASK_TO_SECTION[RESULTS][mask_type]

        # We save the predictions
        targets = targets if targets is not None else zeros(predictions.shape[0])
        if len(predictions.shape) == 0:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PREDICTION: str(predictions[j].item()),
                    TARGET: str(targets[j].item())}
        else:
            for j, id_ in enumerate(ids):
                self._data[section][str(id_)] = {
                    PREDICTION: str(predictions[j].tolist()),
                    TARGET: str(targets[j].item())}

    def record_test_predictions(
            self,
            ids: List[str],
            predictions: Tensor,
            targets: Tensor
    ) -> None:
        """
        Records the test set's predictions.

        Parameters
        ----------
        ids : List[str]
            List of patient/participant ids.
        predictions : Tensor
            Tensor with predicted targets.
        targets : Tensor
            Tensor with ground truth.
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.TEST)

    def record_train_predictions(
            self,
            ids: List[str],
            predictions: Tensor,
            targets: Tensor
    ) -> None:
        """
        Records the training set's predictions.

        Parameters
        ----------
        ids : List[str]
            List of patient/participant ids.
        predictions : Tensor
            Tensor with predicted targets.
        targets : Tensor
            Tensor with ground truth.
        """
        return self.record_predictions(ids, predictions, targets)

    def record_valid_predictions(
            self,
            ids: List[str],
            predictions: Tensor,
            targets: Tensor
    ) -> None:
        """
        Records the validation set's predictions.

        Parameters
        ----------
        ids : List[str]
            List of patient/participant ids.
        predictions : Tensor
            Tensor with predicted targets.
        targets : Tensor
            Tensor with ground truth.
        """
        return self.record_predictions(ids, predictions, targets, mask_type=MaskType.VALID)
