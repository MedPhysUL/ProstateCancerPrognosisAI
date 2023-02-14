"""
    @file:              tuner.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the Tuner classes used for hyperparameter tuning using
                        https://dl.acm.org/doi/10.1145/3377930.3389817.
"""

from os import makedirs
from os.path import join
from time import strftime
from typing import Any, Dict, Optional, Tuple

from optuna import create_study
from optuna.logging import FATAL, set_verbosity
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.pruners import NopPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization import (
    plot_parallel_coordinate,
    plot_param_importances,
    plot_pareto_front,
    plot_optimization_history
)
import ray

from ..metrics.single_task.base import Direction
from .objective import Objective


class Tuner:
    """
    Object in charge of hyperparameter tuning.
    """

    # HYPERPARAMETERS IMPORTANCE SEED
    HP_IMPORTANCE_SEED: int = 42

    # FIGURES NAME
    HPS_IMPORTANCE_FIG: str = "hp_importance.png"
    PARALLEL_COORD_FIG: str = "parallel_coordinates.png"
    PARETO_FRONT_FIG: str = "pareto_front.png"
    OPTIMIZATION_HIST_FIG: str = "optimization_history.png"

    def __init__(
            self,
            n_trials: int,
            path: str,
            study_name: Optional[str] = None,
            objective: Objective = None,
            save_hps_importance: Optional[bool] = False,
            save_parallel_coordinates: Optional[bool] = False,
            save_pareto_front: Optional[bool] = False,
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
        save_pareto_front : Optional[bool]
            Whether we want to plot the pareto front after tuning.
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
        self.save_pareto_front = save_pareto_front
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
        directions = [task.hps_tuning_metric.direction for task in self._objective.dataset.tasks]

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
        for idx, task in enumerate(self._objective.dataset.tasks):
            fig = plot_param_importances(
                self._study,
                evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED),
                target=lambda t: t.values[idx],
                target_name=task.name
            )

            # We save the graph
            fig.write_image(join(self.path, f"{task.name}_{Tuner.HPS_IMPORTANCE_FIG}"))

    def _plot_parallel_coordinates_graph(self) -> None:
        """
        Plots the parallel coordinates graph and save it in an html file.
        """
        # We generate the parallel coordinate graph with optuna
        for idx, task in enumerate(self._objective.dataset.tasks):
            fig = plot_parallel_coordinate(
                self._study,
                target=lambda t: t.values[idx],
                target_name=task.name
            )

            if task.hps_tuning_metric.direction == Direction.MAXIMIZE:
                fig.data[0]["line"].reversescale = False

            # We save the graph
            fig.write_image(join(self.path, f"{task.name}_{Tuner.PARALLEL_COORD_FIG}"))

    def _plot_pareto_front(self) -> None:
        """
        Plots the pareto front.
        """
        fig = plot_pareto_front(
            self._study,
            target_names=[task.name for task in self._objective.dataset.tasks]
        )

        # We save the graph
        fig.write_image(join(self.path, f"{Tuner.PARETO_FRONT_FIG}"))

    def _plot_optimization_history_graph(self) -> None:
        """
        Plots the optimization history graph and save it in a html file.
        """
        # We generate the optimization history graph with optuna
        for idx, task in enumerate(self._objective.dataset.tasks):
            fig = plot_optimization_history(
                self._study,
                target=lambda t: t.values[idx],
                target_name=task.name
            )

            # We save the graph
            fig.write_image(join(self.path, f"{task.name}_{Tuner.OPTIMIZATION_HIST_FIG}"))

    def get_best_trial(self) -> FrozenTrial:
        """
        Retrieves the best trial among all the trials on the pareto front.

        Returns
        -------
        best_trial : FrozenTrial
            Best trial.
        """
        # TODO : Find a way to choose the best hps set among all the sets on the pareto front. For now, we arbitrarily
        #  choose the first in the list.

        return self._study.best_trials[0]

    def get_hps_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the hyperparameter importances.

        Returns
        -------
        hps_importance : Dict[str, Dict[str, float]]
            Hyperparameters importance for each task.
        """
        hps_importance = {}
        for idx, task in enumerate(self._objective.dataset.tasks):
            importances = get_param_importances(
                study=self._study,
                evaluator=FanovaImportanceEvaluator(seed=Tuner.HP_IMPORTANCE_SEED),
                target=lambda t: t.values[idx]
            )

            hps_importance[task.name] = importances

        return hps_importance

    def tune(
            self,
            verbose: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Searches for the hyperparameters that optimize the objective function, using the TPE algorithm.

        Parameters
        ----------
        verbose : bool
            Whether we want optuna to show a progress bar.

        Returns
        -------
        best_hps, hps_importance : Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]
            Best hyperparameters and hyperparameters' importance.
        """
        if self._study is None or self._objective is None:
            raise Exception("study and objective must be defined")

        # We check ray status
        ray_already_init = self._check_ray_status()

        # We perform the optimization
        set_verbosity(FATAL)  # We remove verbosity from loading bar
        self._study.optimize(self._objective, self.n_trials, gc_after_trial=True, show_progress_bar=verbose)

        # We save the plots if it is required
        if self.save_hps_importance:
            self._plot_hps_importance_graph()

        if self.save_parallel_coordinates:
            self._plot_parallel_coordinates_graph()

        if self.save_optimization_history:
            self._plot_optimization_history_graph()

        if self.save_pareto_front:
            self._plot_pareto_front()

        # We extract the best hyperparameters and their importance
        best_trial = self.get_best_trial()
        best_hps = self._objective.extract_hps(best_trial)
        hps_importance = self.get_hps_importance()

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
