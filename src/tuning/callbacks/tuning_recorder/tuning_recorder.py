"""
    @file:              tuning_recorder.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 03/2023

    @Description:       This file is used to define the 'TuningRecorder' callback.
"""

from functools import partial
from itertools import count
from json import dump
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
import torch

from ..base import Priority, TuningCallback
from ....data.datasets import TableDataset
from .json_encoder import EnhancedJSONEncoder
from ....models.torch.base import TorchModel
from ...states import TuningState
from ....visualization import TableViewer


class TuningRecorder(TuningCallback):
    """
    This class is used to manage and create the records of the tuning process.
    """

    instance_counter = count()

    # KEYS
    BEST_TRIAL_KEY: str = "best_trial"
    BEST_TRIALS_KEY: str = "best_trials"
    HISTORY_KEY = "history"
    HYPERPARAMETERS_KEY = "hyperparameters"
    HYPERPARAMETERS_IMPORTANCE_KEY: str = "hyperparameter_importance"
    SCORES_KEY = "scores"
    STATISTICS_KEY = "statistics"
    FIXED_PARAMS = "fixed_params"

    # PREFIXES
    SPLIT_PREFIX: str = "split"
    TRIAL_PREFIX: str = "trial"
    STUDY_PREFIX: str = "study"

    # FOLDERS NAME
    BEST_MODEL_FOLDER_NAME: str = "best_model"
    DESCRIPTIVE_ANALYSIS_FOLDER_NAME: str = "descriptive_analysis"
    HYPERPARAMETERS_FOLDER_NAME: str = "hyperparameters"
    FIGURES_FOLDER_NAME: str = "figures"
    INNER_LOOPS_FOLDER_NAME: str = "inner_splits"
    OUTER_LOOPS_FOLDER_NAME: str = "outer_splits"
    TRIALS_RECORDS_FOLDER_NAME: str = "trials"

    # FILES NAME
    BEST_HYPERPARAMETERS_FILE_NAME: str = "best_hyperparameters.json"
    SUMMARY_FILE_NAME: str = "summary.json"
    SCORES_FILE_NAME: str = "scores.json"
    TORCH_BEST_MODEL_FILE_NAME: str = "best_model.pt"
    SKLEARN_BEST_MODEL_FILE_NAME: str = "best_model.joblib"

    # HYPERPARAMETERS IMPORTANCE SEED
    HP_IMPORTANCE_SEED: int = 42

    # FIGURES NAME
    HPS_IMPORTANCE_FIGURE_NAME: str = "hps_importance.png"

    def __init__(
            self,
            name: Optional[str] = None,
            path_to_record_folder: str = "./tuning_records",
            save_descriptive_analysis: bool = False,
            verbose: bool = False
    ):
        """
        Initializes the records folder.

        Parameters
        ----------
        name : Optional[str]
            The name of the callback.
        path_to_record_folder : str
            Path to the folder to save the records to.
        save_descriptive_analysis : bool
            Whether to save the descriptive analysis.
        verbose : bool
            Whether to print out the trace of the checkpoint.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name)

        os.makedirs(path_to_record_folder, exist_ok=False)
        self.path_to_record_folder = path_to_record_folder
        self.path_to_trials = None
        self.save_descriptive_analysis = save_descriptive_analysis
        self.verbose = verbose

        self._dump = partial(dump, indent=4, sort_keys=True, cls=EnhancedJSONEncoder)
        self._encode = EnhancedJSONEncoder().default

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'CallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
        """
        return False

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.MEDIUM_PRIORITY

    def on_tuning_start(self, tuner, **kwargs):
        """
        Called when the tuning starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        path_to_outer_loops = os.path.join(self.path_to_record_folder, self.OUTER_LOOPS_FOLDER_NAME)
        os.makedirs(path_to_outer_loops, exist_ok=True)

        path_to_hyperparameters = os.path.join(self.path_to_record_folder, self.HYPERPARAMETERS_FOLDER_NAME)
        os.makedirs(path_to_hyperparameters, exist_ok=True)

    @staticmethod
    def _plot_importance(hps_importance_stats: Dict[str, Any], path: str):
        """
        Creates a bar plot with mean and standard deviations of hyperparameters importance.
        """
        means, stds, labels = [], [], []

        for hp_name, hp_stats in hps_importance_stats.items():
            means.append(hp_stats[TuningState.MEAN])
            stds.append(hp_stats[TuningState.STD])
            labels.append(hp_name)

        # We sort the list according to their values
        sorted_means = sorted(means)
        sorted_labels = sorted(labels, key=lambda x: means[labels.index(x)])
        sorted_stds = sorted(stds, key=lambda x: means[stds.index(x)])

        # We build the plot
        y_pos = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.barh(y_pos, sorted_means, xerr=sorted_stds, capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel('Importance')

        # We save the plot
        plt.savefig(path, dpi=300)
        plt.close()

    def on_tuning_end(self, tuner, **kwargs):
        """
        Called when the tuning ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        path_to_hps = os.path.join(self.path_to_record_folder, self.HYPERPARAMETERS_FOLDER_NAME)
        path_to_figures = os.path.join(path_to_hps, self.FIGURES_FOLDER_NAME)
        os.makedirs(path_to_figures, exist_ok=True)

        for idx, task in enumerate(tuner.outer_loop_state.dataset.tasks):
            path_to_task_figure_folder = os.path.join(path_to_figures, task.name)
            os.makedirs(path_to_task_figure_folder, exist_ok=True)
            path_to_task_figure = os.path.join(path_to_task_figure_folder, self.HPS_IMPORTANCE_FIGURE_NAME)
            self._plot_importance(
                tuner.tuning_state.hyperparameters_importance_statistics[task.name],
                path_to_task_figure
            )

        summary = {
            self.SCORES_KEY: {
                self.HISTORY_KEY: tuner.tuning_state.scores_history,
                self.STATISTICS_KEY: tuner.tuning_state.scores_statistics
            },
            self.HYPERPARAMETERS_KEY: {
                self.HISTORY_KEY: tuner.tuning_state.hyperparameters_history,
                self.STATISTICS_KEY: tuner.tuning_state.hyperparameters_statistics
            },
            self.HYPERPARAMETERS_IMPORTANCE_KEY: {
                self.HISTORY_KEY: tuner.tuning_state.hyperparameters_importance_history,
                self.STATISTICS_KEY: tuner.tuning_state.hyperparameters_importance_statistics
            }
        }
        path_to_summary = os.path.join(path_to_hps, self.SUMMARY_FILE_NAME)
        with open(path_to_summary, "w") as file:
            self._dump(summary, file)

    def on_outer_loop_start(self, tuner, **kwargs):
        """
        Called when the outer loop starts.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        self._set_path_to_outer_loop_folder(tuner)

        self.path_to_trials = os.path.join(
            tuner.outer_loop_state.path_to_outer_loop_folder,
            self.TRIALS_RECORDS_FOLDER_NAME
        )
        os.makedirs(self.path_to_trials, exist_ok=True)

        if self.save_descriptive_analysis:
            table_dataset = tuner.outer_loop_state.dataset.table_dataset
            if isinstance(table_dataset, TableDataset):
                table_viewer = TableViewer(table_dataset)
                table_viewer.visualize(
                    path_to_save=os.path.join(
                        tuner.outer_loop_state.path_to_outer_loop_folder,
                        self.DESCRIPTIVE_ANALYSIS_FOLDER_NAME
                    )
                )

    def _set_path_to_outer_loop_folder(self, tuner):
        """
        Sets path to outer loop folder.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        path_to_outer_loop_folder = os.path.join(
            self.path_to_record_folder,
            self.OUTER_LOOPS_FOLDER_NAME,
            f"{self.SPLIT_PREFIX}_{tuner.outer_loop_state.idx}"
        )
        os.makedirs(path_to_outer_loop_folder, exist_ok=True)
        tuner.outer_loop_state.path_to_outer_loop_folder = path_to_outer_loop_folder

    def on_best_model_evaluation_start(self, tuner, **kwargs):
        """
        Called when the model evaluation starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        self._set_path_to_best_model_folder(tuner)

    def _set_path_to_best_model_folder(self, tuner):
        """
        Sets path to best model folder.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        path_to_best_model_folder = os.path.join(
            tuner.outer_loop_state.path_to_outer_loop_folder,
            self.BEST_MODEL_FOLDER_NAME
        )
        os.makedirs(path_to_best_model_folder, exist_ok=True)
        tuner.best_model_state.path_to_best_model_folder = path_to_best_model_folder

    def on_best_model_evaluation_end(self, tuner, **kwargs):
        """
        Called when the model evaluation ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        path_to_scores = os.path.join(tuner.best_model_state.path_to_best_model_folder, self.SCORES_FILE_NAME)
        with open(path_to_scores, "w") as file:
            self._dump(tuner.best_model_state.score, file)

        if tuner.tuning_state.scores:
            tuner.tuning_state.scores.append(tuner.best_model_state.score)
        else:
            tuner.tuning_state.scores = [tuner.best_model_state.score]

        model = tuner.best_model_state.model
        if isinstance(model, TorchModel):
            model_state = model.state_dict()
            path_to_model = os.path.join(
                tuner.best_model_state.path_to_best_model_folder,
                self.TORCH_BEST_MODEL_FILE_NAME
            )
            torch.save(model_state, path_to_model)
        else:
            raise NotImplementedError

    def on_study_end(self, tuner, **kwargs):
        """
        Called when the study ends.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        path_to_best_hps_file = os.path.join(
            tuner.outer_loop_state.path_to_outer_loop_folder,
            self.BEST_HYPERPARAMETERS_FILE_NAME
        )

        hps_importance = self._get_hps_importance(tuner)
        best_hps_summary = {
            self.STUDY_PREFIX: tuner.study_state.study,
            self.BEST_TRIAL_KEY: self._encode(tuner.study_state.best_trial),
            self.BEST_TRIALS_KEY: self._encode(tuner.study_state.study.best_trials),
            self.HYPERPARAMETERS_IMPORTANCE_KEY: hps_importance
        }

        with open(path_to_best_hps_file, "w") as file:
            self._dump(best_hps_summary, file)

        if tuner.tuning_state.hyperparameters_importance:
            tuner.tuning_state.hyperparameters_importance.append(hps_importance)
        else:
            tuner.tuning_state.hyperparameters_importance = [hps_importance]

        if tuner.tuning_state.hyperparameters:
            tuner.tuning_state.hyperparameters.append(tuner.study_state.best_trial.params)
        else:
            tuner.tuning_state.hyperparameters = [tuner.study_state.best_trial.params]

    def _get_hps_importance(self, tuner):
        """
        Gets the hyperparameter importances.

        Returns
        -------
        hps_importance : Dict[str, Dict[str, float]]
            Hyperparameters importance for each task.
        """
        hps_importance = {}
        for idx, task in enumerate(tuner.outer_loop_state.dataset.tasks):
            importances = get_param_importances(
                study=tuner.study_state.study,
                evaluator=FanovaImportanceEvaluator(seed=self.HP_IMPORTANCE_SEED),
                target=lambda t: t.values[idx]
            )

            hps_importance[task.name] = importances

        return hps_importance

    def on_trial_start(self, objective, **kwargs):
        """
        Called when the trial starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        self._set_path_to_trial_folder(objective)
        path_to_inner_loops = os.path.join(
            objective.trial_state.path_to_trial_folder,
            self.INNER_LOOPS_FOLDER_NAME
        )
        os.makedirs(path_to_inner_loops, exist_ok=True)

    def _set_path_to_trial_folder(self, objective):
        """
        Sets path to trial folder.

        Parameters
        ----------
        objective : Objective
            The objective.
        """
        path_to_trial_folder = os.path.join(
            self.path_to_trials,
            f"{self.TRIAL_PREFIX}_{objective.trial_state.trial.number}"
        )
        os.makedirs(path_to_trial_folder, exist_ok=True)
        objective.trial_state.path_to_trial_folder = path_to_trial_folder

    def on_trial_end(self, objective, **kwargs):
        """
        Called when the trial ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        fixed_hps = {hp.name: hp.value for hp in objective.hyperparameters.fixed_hyperparameters}
        objective.trial_state.trial.set_user_attr(self.FIXED_PARAMS, fixed_hps)

        trial_summary = {
            self.TRIAL_PREFIX: objective.trial_state.trial,
            self.HISTORY_KEY: self._encode(objective.trial_state.history),
            self.STATISTICS_KEY: self._encode(objective.trial_state.statistics)
        }

        path_to_hps = os.path.join(objective.trial_state.path_to_trial_folder, self.SUMMARY_FILE_NAME)
        with open(path_to_hps, "w") as file:
            self._dump(trial_summary, file)

    def on_inner_loop_start(self, objective, **kwargs):
        """
        Called when the inner loop starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        self._set_path_to_inner_loop_folder(objective)

        if self.save_descriptive_analysis:
            table_dataset = objective.inner_loop_state.dataset.table_dataset
            if isinstance(table_dataset, TableDataset):
                table_viewer = TableViewer(table_dataset)
                table_viewer.visualize(
                    path_to_save=os.path.join(
                        objective.inner_loop_state.path_to_inner_loop_folder,
                        self.DESCRIPTIVE_ANALYSIS_FOLDER_NAME
                    )
                )

    def _set_path_to_inner_loop_folder(self, objective):
        """
        Sets path to inner loop folder.

        Parameters
        ----------
        objective : Objective
            The objective.
        """
        path_to_inner_loop_folder = os.path.join(
            objective.trial_state.path_to_trial_folder,
            self.INNER_LOOPS_FOLDER_NAME,
            f"{self.SPLIT_PREFIX}_{objective.inner_loop_state.idx}"
        )
        os.makedirs(path_to_inner_loop_folder, exist_ok=True)
        objective.inner_loop_state.path_to_inner_loop_folder = path_to_inner_loop_folder

    def on_inner_loop_end(self, objective, **kwargs):
        """
        Called when the inner loop ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        path_to_scores = os.path.join(objective.inner_loop_state.path_to_inner_loop_folder, self.SCORES_FILE_NAME)
        with open(path_to_scores, "w") as file:
            self._dump(objective.inner_loop_state.score, file)
