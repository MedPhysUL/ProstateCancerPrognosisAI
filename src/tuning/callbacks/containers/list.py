"""
    @file:              list.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the 'TuningCallbackList' class which essentially acts as a list of
                        callbacks.
"""

from typing import Optional, Iterable, Iterator

from ..base import TuningCallback


class TuningCallbackList:
    """
    Holds tuning callbacks in a list. Each callback in the list is called in the order it is stored in the list.
    """

    def __init__(self, callbacks: Optional[Iterable[TuningCallback]] = None):
        """
        Constructor of the Callbacks class.

        Parameters
        ----------
        callbacks : Iterable[TuningCallback]
            The callbacks to use.
        """
        if callbacks is None:
            callbacks = []
        assert isinstance(callbacks, Iterable), "callbacks must be an Iterable."
        assert all(isinstance(callback, TuningCallback) for callback in callbacks), (
            "All callbacks must be instances of TuningCallback."
        )

        self.callbacks = list(callbacks)
        self.check_for_duplicate_callback_classes()
        self.check_for_duplicate_callback_names()
        self.sort()

    def __getitem__(self, idx: int) -> TuningCallback:
        """
        Get a callback from the list.

        Parameters
        ----------
        idx : int
            The index of the callback to get.

        Returns
        -------
        callback : Callback
            The callback at the given index in the list of callbacks.
        """
        return self.callbacks[idx]

    def __iter__(self) -> Iterator[TuningCallback]:
        """
        Get an iterator over the callbacks.

        Returns
        -------
        iterator : Iterator[TuningCallback]
            An iterator over the callbacks.
        """
        return iter(self.callbacks)

    def __len__(self) -> int:
        """
        Get the number of callbacks in the list.

        Returns
        -------
        number : int
            The number of callbacks in the list.
        """
        return len(self.callbacks)

    def append(self, callback: TuningCallback):
        """
        Append a callback to the list.

        Parameters
        ----------
        callback : TuningCallback
            The callback to append.
        """
        assert isinstance(callback, TuningCallback), "callback must be an instance of TuningCallback"
        self.callbacks.append(callback)
        self.check_for_duplicate_callback_classes()
        self.check_for_duplicate_callback_names()
        self.sort()

    def check_for_duplicate_callback_classes(self):
        """
        Checks that the duplicated classes in the CallbackList are allowed duplicates.
        """
        seen, duplicates = [], []

        for callback in self.callbacks:
            if callback.allow_duplicates is False:
                if isinstance(callback, tuple(seen)):
                    duplicates.append(callback.__class__.__name__)
                else:
                    seen.append(type(callback))

        if duplicates:
            raise AssertionError(f"Duplicates of 'TuningCallback' with 'allow_duplicates' == False are not allowed in "
                                 f"'TuningCallbackList'. Found duplicates {duplicates}.")

    def check_for_duplicate_callback_names(self):
        """
        Check if there is any duplicate callback names in the CallbackList.
        """
        seen, duplicates = [], []

        for callback in self.callbacks:
            if callback.name in seen:
                duplicates.append(callback.name)
            else:
                seen.append(callback.name)

        if duplicates:
            raise AssertionError(
                f"Duplicates callback names are not allowed in 'TuningCallbackList'. Found duplicates {duplicates}."
            )

    def remove(self, callback: TuningCallback):
        """
        Remove a callback from the list.

        Parameters
        ----------
        callback : TuningCallback
            The callback to remove.
        """
        assert isinstance(callback, TuningCallback), "callback must be an instance of TuningCallback"
        self.callbacks.remove(callback)
        self.sort()

    def sort(self):
        """
        Sorts the callbacks using their priority.
        """
        self.callbacks.sort(key=lambda callback: callback.priority, reverse=True)

    def on_tuning_start(self, tuner, **kwargs):
        """
        Called when the tuning starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_tuning_start(tuner, **kwargs)

    def on_tuning_end(self, tuner, **kwargs):
        """
        Called when the tuning ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_tuning_end(tuner, **kwargs)

    def on_outer_loop_start(self, tuner, **kwargs):
        """
        Called when the outer loop starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_outer_loop_start(tuner, **kwargs)

    def on_outer_loop_end(self, tuner, **kwargs):
        """
        Called when the outer loop ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_outer_loop_end(tuner, **kwargs)

    def on_best_model_evaluation_start(self, objective, **kwargs):
        """
        Called when the model model evaluation starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        for callback in self.callbacks:
            callback.on_best_model_evaluation_start(objective, **kwargs)

    def on_best_model_evaluation_end(self, objective, **kwargs):
        """
        Called when the model model evaluation ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        for callback in self.callbacks:
            callback.on_best_model_evaluation_start(objective, **kwargs)

    def on_study_start(self, tuner, **kwargs):
        """
        Called when the study starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_study_start(tuner, **kwargs)

    def on_study_end(self, tuner, **kwargs):
        """
        Called when the study ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        for callback in self.callbacks:
            callback.on_study_end(tuner, **kwargs)

    def on_trial_start(self, objective, **kwargs):
        """
        Called when the trial starts.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        for callback in self.callbacks:
            callback.on_trial_start(objective, **kwargs)

    def on_trial_end(self, objective, **kwargs):
        """
        Called when the trial ends.

        Parameters
        ----------
        objective : Objective
            Objective.
        """
        for callback in self.callbacks:
            callback.on_trial_end(objective, **kwargs)

    def on_inner_loop_start(self, objective, **kwargs):
        """
        Called when the inner loop starts.

        Parameters
        ----------
        objective : Objective
            The objective.
        """
        for callback in self.callbacks:
            callback.on_inner_loop_start(objective, **kwargs)

    def on_inner_loop_end(self, objective, **kwargs):
        """
        Called when the inner loop ends.

        Parameters
        ----------
        objective : Objective
            The objective.
        """
        for callback in self.callbacks:
            callback.on_inner_loop_end(objective, **kwargs)
