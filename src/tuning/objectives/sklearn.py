"""
    @file:              sklearn.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the `SklearnObjective` class used for hyperparameters tuning.
"""

from typing import Any, Dict

from .base import ModelEvaluationContainer, Objective
from ...data.datasets import ProstateCancerDataset
from ..hyperparameters.containers import HyperparameterDict
from ..hyperparameters.sklearn import FitMethodHyperparameter, SklearnModelHyperparameter


class SklearnObjective(Objective):
    """
    Callable objective function to use with the tuner for SklearnModel.
    """

    MODEL_INSTANCE_KEY = "model"
    FIT_METHOD_PARAMS_KEY = "fit"

    def __init__(
            self,
            model_hyperparameter: SklearnModelHyperparameter,
            fit_method_hyperparameter: FitMethodHyperparameter
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        model_hyperparameter : SklearnModelHyperparameter
            Model constructor hyperparameters.
        fit_method_hyperparameter : FitMethodHyperparameter
            Fit method hyperparameters.
        """
        super().__init__()

        self._hyperparameters = HyperparameterDict(
            {
                self.MODEL_INSTANCE_KEY: model_hyperparameter,
                self.FIT_METHOD_PARAMS_KEY: fit_method_hyperparameter
            }
        )

    @property
    def hyperparameters(self) -> HyperparameterDict:
        """
        Hyperparameters.

        Returns
        -------
        hyperparameters : HyperparameterDict
            Dictionary containing hyperparameters.
        """
        return self._hyperparameters

    def _test_hyperparameters(
            self,
            dataset: ProstateCancerDataset,
            hyperparameters: Dict[str, Any],
            path_to_save: str
    ) -> ModelEvaluationContainer:
        """
        Tests hyperparameters and returns the train, valid and test scores.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            The dataset used for the current trial.
        hyperparameters : Dict[str, Any]
            Suggested hyperparameters for this trial.
        path_to_save : str
            Path to the directory to save the current scores.

        Returns
        -------
        model_evaluation : ModelEvaluationContainer
            Model evaluation.
        """
        # We retrieve the model instance and the fit method parameters from the suggested hyperparameters
        model_instance = hyperparameters[self.MODEL_INSTANCE_KEY]
        fit_method_params = hyperparameters[self.FIT_METHOD_PARAMS_KEY]
        dataset = self.inner_loop_state.dataset
        model_instance.build(dataset)
        fit_method_params[self.DATASET_KEY] = dataset

        # We train the model using the suggested hyperparameters
        model_instance.fit(**fit_method_params)

        return self._get_model_evaluation(model_instance, dataset)
