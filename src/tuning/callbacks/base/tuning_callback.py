"""
    @file:              tuning_callback.py
    @Author:            Maxence Larose

    @Creation Date:     10/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the TuningCallback abstract class.
"""

from abc import ABC, abstractmethod


class TuningCallback(ABC):
    """
    Abstract base class used to build tuning callbacks. A callback is a self-contained program that can be used to
    monitor or alter the tuning process. The advantage of callbacks is that it decouples a lot of non-essential logic
    that does not need to live in the tuner.

    Nomenclature:
        - Loop: The process of looping through the dataset in subsets for tuning (nested cross-validation).
        - Study : An optimization task, i.e., a set of trials.
        - Trial: The process of evaluating an objective function with a given set of hyperparameters.
        - Tuning: The process of selecting the optimal set of hyperparameters.

    Callback methods are called in the following order:
        - `on_tuning_start`
        * Executes n_outer_loops times :
            - `on_outer_loop_start`
            - `on_study_start`
            * Executes n_trials times:
                - `on_trial_start`
                * Executes n_inner_loops_times (in parallel if possible):
                    - `on_inner_loop_start`
                    - " MODEL TRAINING "
                - `on_trial_end`
            - `on_study_end`
            - `on_best_model_evaluation_start`
            - `on_best_model_evaluation_end`
            - `on_outer_loop_end`
        - `on_tuning_end`

    Complete list of the training and tuning hooks:
        - `on_tuning_start`
        * Executes n_outer_loops times :
            - `on_outer_loop_start`
            - `on_study_start`
            * Executes n_trials times:
                - `on_trial_start`
                * Executes n_inner_loops_times (in parallel if possible):
                    - `on_inner_loop_start`
                    - `on_fit_start`
                    * Executes n_epochs times:
                        - `on_epoch_start`
                        - `on_train_start`
                        * Executes n_train_batches times:
                            - `on_train_batch_start`
                            - `on_optimization_start`
                            - `on_optimization_end`
                            - `on_train_batch_end`
                        - `on_train_end`
                        - `on_validation_start`
                        * Executes n_valid_batches times:
                            - `on_validation_batch_start`
                            - `on_validation_batch_end`
                        - `on_validation_end`
                        - `on_epoch_end`
                    - `on_inner_loop_end`
                    - `on_fit_end`
                - `on_trial_end`
            - `on_study_end`
            - `on_best_model_evaluation_start`
            - `on_best_model_evaluation_end`
            - `on_outer_loop_end`
        - `on_tuning_end`

    """

    UNSERIALIZABLE_ATTRIBUTES = ["tuner"]

    def __init__(
            self,
            name: str
    ):
        """
        Initialize name and state of the callback.

        Parameters
        ----------
        name : str
            The name of the callback.
        """
        self.name = name
        self.tuner = None

    @property
    @abstractmethod
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific TuningCallback class in the 'TuningCallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority : int
            Callback priority.
        """
        raise NotImplementedError

    def on_tuning_start(self, tuner, **kwargs):
        """
        Called when the tuning starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        self.tuner = tuner

    def on_tuning_end(self, tuner, **kwargs):
        """
        Called when the tuning ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        pass

    def on_outer_loop_start(self, tuner, **kwargs):
        """
        Called when the outer loop starts.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_outer_loop_end(self, tuner, **kwargs):
        """
        Called when the outer loop ends.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_best_model_evaluation_start(self, tuner, **kwargs):
        """
        Called when the model model evaluation starts.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_best_model_evaluation_end(self, tuner, **kwargs):
        """
        Called when the model model evaluation ends.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_study_start(self, tuner, **kwargs):
        """
        Called when the study starts.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_study_end(self, tuner, **kwargs):
        """
        Called when the study ends.

        Parameters
        ----------
        tuner : Tuner
            Tuner.
        """
        pass

    def on_trial_start(self, objective, **kwargs):
        """
        Called when the trial starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        pass

    def on_trial_end(self, objective, **kwargs):
        """
        Called when the trial ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        pass

    def on_inner_loop_start(self, objective, **kwargs):
        """
        Called when the inner loop starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        pass

    def on_inner_loop_end(self, objective, **kwargs):
        """
        Called when the inner loop ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        pass
