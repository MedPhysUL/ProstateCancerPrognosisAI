"""
    @file:              objective.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the Objective class used for hyperparameter tuning.
"""

from copy import deepcopy
from os import cpu_count
from typing import Any, Callable, Dict, List, Optional, Union

from optuna.trial import FrozenTrial, Trial
import ray
from torch import mean, tensor

from ..data.processing.sampling import Mask
from ..data.datasets import ProstateCancerDataset
from .hyperparameters import (
    CategoricalHyperparameter,
    Distribution,
    Hyperparameter,
    NumericalContinuousHyperparameter,
    NumericalIntHyperparameter,
    Range
)
from ..models.base.base_model import BaseModel


class Objective:
    """
    Callable objective function to use with the tuner.
    """

    def __init__(
            self,
            dataset: ProstateCancerDataset,
            masks: Dict[int, Dict[str, List[int]]],
            hps: Dict[str, Dict[str, Any]],
            fixed_params: Optional[Dict[str, Any]],
            model_constructor: BaseModel,
            gpu_device: bool = False
    ) -> None:
        """
        Sets protected and public attributes of the objective.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Custom dataset containing all the data needed for our evaluations.
        masks : Dict[int, Dict[str, List[int]]]
            Dict with list of idx to use as train, valid and test masks.
        hps : Dict[str, Dict[str, Any]]
            Dictionary with information on the hyperparameters we want to tune.
        model_constructor : BaseModel
            Callable object that builds a model using hyperparameters and fixed params.
        gpu_device : bool
            Whether we want to use a GPU.
        """
        # We validate the given hyperparameters
        for hp in model_constructor.get_hyperparameters():
            if not (hp.name in list(hps.keys())):
                raise ValueError(f"'{hp}' is missing from hps dictionary")

        # We set protected attributes
        self._dataset = dataset
        self._fixed_params = fixed_params if fixed_params is not None else {}
        self._hps = hps
        self._masks = masks
        self._model_constructor = model_constructor
        self._getters = {}
        self._define_getters()

        # We set the protected parallel evaluation method
        self._run_single_evaluation = self._build_parallel_process(gpu_device)

    @property
    def dataset(self) -> ProstateCancerDataset:
        return self._dataset

    def __call__(
            self,
            trial: Trial
    ) -> float:
        """
        Extracts hyperparameters suggested by optuna and executes the parallel evaluations of the hyperparameters set.

        Parameters
        ----------
        trial : Trial
            Optuna trial.

        Returns
        -------
        score : float
            Score associated to the set of hyperparameters.
        """
        # We extract hyperparameters suggestions
        suggested_hps = {k: f(trial) for k, f in self._getters.items()}

        # We execute parallel evaluations
        futures = [self._run_single_evaluation.remote(masks=m, hps=suggested_hps) for k, m in self._masks.items()]
        scores = ray.get(futures)

        # We take the mean of the scores
        return mean(tensor(scores), dim=0).tolist()

    def _define_getters(
            self
    ) -> None:
        """
        Defines the different optuna sampling function for each hyperparameter.
        """
        # For each hyperparameter associated to the model we are tuning
        for hp in self._model_constructor.get_hyperparameters():

            # We check if a value was predefined for this hyperparameter,
            # in this case, no value we'll be sampled by Optuna
            if Range.VALUE in self._hps[hp.name].keys():
                self._getters[hp.name] = self._build_constant_getter(hp)

            # Otherwise we build the suggestion function appropriate to the hyperparameter
            elif hp.distribution == Distribution.CATEGORICAL:
                self._getters[hp.name] = self._build_categorical_getter(hp)

            elif hp.distribution == Distribution.UNIFORM:
                self._getters[hp.name] = self._build_numerical_cont_getter(hp)

            else:  # discrete uniform distribution
                self._getters[hp.name] = self._build_numerical_int_getter(hp)

    def _build_constant_getter(
            self,
            hp: Hyperparameter
    ) -> Callable:
        """
        Builds a function that extracts the given predefined hyperparameter value.

        Parameters
        ----------
        hp : Hyperparameter
            hyperparameter

        Returns
        -------
        function : Callable
            Constant getter
        """
        def getter(trial: Trial) -> Any:
            return self._hps[hp.name][Range.VALUE]

        return getter

    def _build_categorical_getter(
            self,
            hp: CategoricalHyperparameter
    ) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the categorical hyperparameter.

        Parameters
        ----------
        hp : CategoricalHyperparameter
            Categorical hyperparameter.

        Returns
        -------
        function : Callable
            Categorical getter
        """
        def getter(trial: Trial) -> str:
            return trial.suggest_categorical(hp.name, self._hps[hp.name][Range.VALUES])

        return getter

    def _build_numerical_int_getter(
            self,
            hp: NumericalIntHyperparameter
    ) -> Callable:
        """
         Builds a function that extracts optuna's suggestion for the numerical discrete hyperparameter.

        Parameters
        ----------
        hp : NumericalIntHyperparameter
            Numerical discrete hyperparameter.

        Returns
        -------
        function : Callable
            Categorical getter
        """
        def getter(trial: Trial) -> int:
            return trial.suggest_int(hp.name, self._hps[hp.name][Range.MIN], self._hps[hp.name][Range.MAX],
                                     step=self._hps[hp.name].get(Range.STEP, 1))
        return getter

    def _build_numerical_cont_getter(
            self,
            hp: NumericalContinuousHyperparameter
    ) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the numerical continuous hyperparameter.

        Parameters
        ----------
        hp : NumericalContinuousHyperparameter
            Numerical continuous hyperparameter.

        Returns
        -------
        function : Callable
            Categorical getter
        """
        def getter(trial: Trial) -> Union[float]:
            return trial.suggest_uniform(hp.name, self._hps[hp.name][Range.MIN], self._hps[hp.name][Range.MAX])

        return getter

    def _build_parallel_process(
            self,
            gpu_device: bool = False
    ) -> Callable:
        """
        Builds the function run in parallel for each set of hyperparameters and return the score.

        Parameters
        ----------
        gpu_device : bool
            Whether we want to use a gpu.

        Returns
        -------
        run_function : Callable
            Function that train a single model using given masks and given HPs.
        """

        # TODO : num_cpus and num_gpus should be user selectable. For now, we have to use all cpus per trial,
        #  otherwise Ray thinks the tasks have to be parallelized and everything falls apart because the GPU memory is
        #  not big enough to hold 2 trials (models) at a time.
        @ray.remote(num_cpus=cpu_count(), num_gpus=int(gpu_device))
        def run_single_evaluation(
                masks: Dict[str, List[int]],
                hps: Dict[str, Any]
        ) -> List[float]:
            """
            Train a single model using given masks and given hyperparameters.

            Parameters
            ----------
            masks : Dict[str, List[int]]
                Dictionary with list of integers for train, valid and test mask.
            hps : Dict[str, Any]
                Dictionary with hyperparameters to give to the model constructor.

            Returns
            -------
            scores : List[float]
                List of score values.
            """
            # We extract masks
            train_idx, valid_idx, test_idx = masks[Mask.TRAIN.value], masks[Mask.VALID.value], masks[Mask.TEST.value]

            # We create a copy of the current dataset and update its masks
            dts = deepcopy(self._dataset)
            dts.update_masks(train_mask=train_idx, valid_mask=valid_idx, test_mask=test_idx)

            # We build a model using hps and fixed params (BaseModel)
            model = self._model_constructor(**hps, **self._fixed_params)

            # We train the model
            model.fit(dts)

            # We find the optimal threshold for each classification tasks
            model.fix_thresholds_to_optimal_values(dts)

            # We calculate the scores on the different tasks
            test_set_scores = model.scores_dataset(dataset=dts, mask=dts.test_mask)

            # We retrieve the score associated to the optimization metric
            scores = [test_set_scores[task.name][task.hps_tuning_metric.name] for task in dts.tasks]

            return scores

        return run_single_evaluation

    def extract_hps(
            self,
            trial: FrozenTrial
    ) -> Dict[str, Any]:
        """
        Extracts model hyperparameters in a dictionary with the appropriate keys given an optuna trial.

        Parameters
        ----------
        trial : FrozenTrial
            Optuna frozen trial.

        Returns
        -------
        dictionary : Dict[str, Any]
            Dictionary with hyperparameters' values.
        """
        return {hp.name: self._hps[hp.name].get(Range.VALUE, trial.params.get(hp.name))
                for hp in self._model_constructor.get_hyperparameters()}
