"""
    @file:              tuner.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 08/2022

    @Description:       This file is used to define the Objective and Tuner classes used for hyperparameter tuning
                        using https://dl.acm.org/doi/10.1145/3377930.3389817.
"""

from copy import deepcopy
from os import makedirs
from os.path import join
from time import strftime
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from optuna import create_study
from optuna.logging import FATAL, set_verbosity
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_optimization_history
import ray
from torch import mean, tensor

from src.data.datasets.table_dataset import MaskType
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.models.base.base_model import BaseModel
from src.utils.hyperparameters import CategoricalHP, Distribution, HP, NumericalContinuousHP, NumericalIntHP, Range


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
        for hp in model_constructor.get_hps():
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
        return mean(tensor(scores)).item()

    def _define_getters(
            self
    ) -> None:
        """
        Defines the different optuna sampling function for each hyperparameter.
        """
        # For each hyperparameter associated to the model we are tuning
        for hp in self._model_constructor.get_hps():

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
            hp: HP
    ) -> Callable:
        """
        Builds a function that extracts the given predefined hyperparameter value.

        Parameters
        ----------
        hp : HP
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
            hp: CategoricalHP
    ) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the categorical hyperparameter.

        Parameters
        ----------
        hp : CategoricalHP
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
            hp: NumericalIntHP
    ) -> Callable:
        """
         Builds a function that extracts optuna's suggestion for the numerical discrete hyperparameter.

        Parameters
        ----------
        hp : NumericalIntHP
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
            hp: NumericalContinuousHP
    ) -> Callable:
        """
        Builds a function that extracts optuna's suggestion for the numerical continuous hyperparameter.

        Parameters
        ----------
        hp : NumericalContinuousHP
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

        @ray.remote(num_gpus=int(gpu_device))
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
            train_idx, valid_idx, test_idx = masks[MaskType.TRAIN], masks[MaskType.VALID], masks[MaskType.TEST]

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
            test_set_scores = model.score_dataset(dataset=dts, mask=dts.test_mask)
            scores = list(test_set_scores.values())

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
                for hp in self._model_constructor.get_hps()}


class Tuner:
    """
    Object in charge of hyperparameter tuning.
    """

    # HYPERPARAMETERS IMPORTANCE SEED
    HP_IMPORTANCE_SEED: int = 42

    # FIGURES NAME
    HPS_IMPORTANCE_FIG: str = "hp_importance.png"
    PARALLEL_COORD_FIG: str = "parallel_coordinates.png"
    OPTIMIZATION_HIST_FIG: str = "optimization_history.png"

    def __init__(
            self,
            n_trials: int,
            path: str,
            study_name: Optional[str] = None,
            objective: Objective = None,
            save_hps_importance: Optional[bool] = False,
            save_parallel_coordinates: Optional[bool] = False,
            save_optimization_history: Optional[bool] = False
    ):
        """
        Sets all protected and public attributes.

        Parameters
        ----------
        n_trials : int
            Number of sets of hyperparameters tested.
        path : str
            Path of the directory used to store graphs created.
        study_name : Optional[str]
            Name of the optuna study.
        objective : Objective
            Objective function to optimize.
        save_hps_importance : Optional[bool]
            Whether we want to plot the hyperparameters importance graph after tuning.
        save_parallel_coordinates : Optional[bool]
            Whether we want to plot the parallel coordinates graph after tuning.
        save_optimization_history : Optional[bool]
            Whether we want to plot the optimization history graph after tuning.
        """

        # We set protected attributes
        self._objective = objective
        self._study = self._new_study(study_name) if study_name is not None else None

        # We set public attributes
        self.n_trials = n_trials
        self.path = join(path, f"{strftime('%Y%m%d-%H%M%S')}")
        self.save_hps_importance = save_hps_importance
        self.save_parallel_coordinates = save_parallel_coordinates
        self.save_optimization_history = save_optimization_history

        # We make sure that the path given exists
        makedirs(self.path, exist_ok=True)

    def _new_study(
            self,
            study_name: str
    ) -> Study:
        """
        Creates a new optuna study.

        Parameters
        ----------
        study_name : str
            Name of the optuna study.

        Returns
        -------
        study : Study
            Study object.
        """
        directions = [task.optimization_metric.direction for task in self._objective.dataset.tasks]

        study = create_study(
            directions=directions,
            study_name=study_name,
            sampler=TPESampler(
                n_startup_trials=20,
                n_ei_candidates=20,
                multivariate=True,
                constant_liar=True
            ),
            pruner=NopPruner()
        )

        return study

    def _plot_hps_importance_graph(self) -> None:
        """
        Plots the hyperparameters importance graph and save it in an html file.
        """
        # We generate the hyperparameters importance graph with optuna
        fig = plot_param_importances(self._study, evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED))

        # We save the graph
        fig.write_image(join(self.path, Tuner.HPS_IMPORTANCE_FIG))

    def _plot_parallel_coordinates_graph(self) -> None:
        """
        Plots the parallel coordinates graph and save it in an html file.
        """
        # We generate the parallel coordinate graph with optuna
        fig = plot_parallel_coordinate(self._study)

        # We save the graph
        fig.write_image(join(self.path, Tuner.PARALLEL_COORD_FIG))

    def _plot_optimization_history_graph(self) -> None:
        """
        Plots the optimization history graph and save it in a html file.
        """
        # We generate the optimization history graph with optuna
        fig = plot_optimization_history(self._study)

        # We save the graph
        fig.write_image(join(self.path, Tuner.OPTIMIZATION_HIST_FIG))

    def get_best_hps(self) -> Dict[str, Any]:
        """
        Retrieves the best hyperparameters found in the tuning.

        Returns
        -------
        dictionary : Dict[str, Any]
            Dictionary with hyperparameters' values.
        """
        return self._objective.extract_hps(self._study.best_trial)

    def tune(
            self,
            verbose: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Searches for the hyperparameters that optimize the objective function, using the TPE algorithm.

        Parameters
        ----------
        verbose : bool
            Whether we want optuna to show a progress bar.

        Returns
        -------
        best_hps, hps_importance : Tuple[Dict[str, Any], Dict[str, float]]
            Best hyperparameters and hyperparameters' importance.
        """
        if self._study is None or self._objective is None:
            raise Exception("study and objective must be defined")

        # We check ray status
        ray_already_init = self._check_ray_status()

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self._study.optimize(self._objective, self.n_trials, show_progress_bar=verbose)

        # We save the plots if it is required
        if self.save_hps_importance:
            self._plot_hps_importance_graph()

        if self.save_parallel_coordinates:
            self._plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self._plot_optimization_history_graph()

        # We extract the best hyperparameters and their importance
        best_hps = self.get_best_hps()
        hps_importance = get_param_importances(
            study=self._study,
            evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED)
        )

        # We shutdown ray if it has been initialized in this function
        if not ray_already_init:
            ray.shutdown()

        return best_hps, hps_importance

    def update_tuner(
            self,
            study_name: str,
            objective: Objective,
            saving_path: Optional[str] = None
    ) -> None:
        """
        Sets study and objective protected attributes.

        Parameters
        ----------
        study_name : str
            Name of the optuna study.
        objective : Objective
            Objective function to optimize.
        saving_path : Optional[str]
            Path where the tuning details will be stored.
        """
        self._objective = objective
        self._study = self._new_study(study_name)
        self.path = saving_path if saving_path is not None else self.path

    @staticmethod
    def _check_ray_status() -> bool:
        """
        Checks if ray was already initialized and initialize it if it's not.

        Returns
        -------
        Whether ray was already initialized.
        """
        # We initialize ray if it is not initialized yet
        ray_was_init = True
        if not ray.is_initialized():
            ray_was_init = False
            ray.init()

        return ray_was_init
