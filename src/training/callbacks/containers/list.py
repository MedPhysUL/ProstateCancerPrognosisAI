"""
    @file:              list.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the 'TrainingCallbackList' class which essentially acts as a list of
                        callbacks. A lot of the logic behind the following code is borrowed from PyTorch Lightning
                        (https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) and
                        NeuroTorch (https://github.com/NeuroTorch/NeuroTorch).
"""

from typing import Any, Dict, Optional, Iterable, Iterator

from ..base import TrainingCallback


class TrainingCallbackList:
    """
    Holds callbacks in a list. Each callback in the list is called in the order it is stored in the list.
    """

    def __init__(self, callbacks: Optional[Iterable[TrainingCallback]] = None):
        """
        Constructor of the TrainingCallbackList class.

        Parameters
        ----------
        callbacks : Iterable[TrainingCallback]
            The callbacks to use.
        """
        if callbacks is None:
            callbacks = []
        assert isinstance(callbacks, Iterable), "callbacks must be an Iterable."
        assert all(isinstance(callback, TrainingCallback) for callback in callbacks), (
            "All callbacks must be instances of TrainingCallback."
        )

        self.callbacks = list(callbacks)
        self.check_for_duplicate_callback_classes()
        self.check_for_duplicate_callback_names()
        self.sort()

    def __getitem__(self, idx: int) -> TrainingCallback:
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

    def __iter__(self) -> Iterator[TrainingCallback]:
        """
        Get an iterator over the callbacks.

        Returns
        -------
        iterator : Iterator[TrainingCallback]
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

    def append(self, callback: TrainingCallback):
        """
        Append a callback to the list.

        Parameters
        ----------
        callback : Callback
            The callback to append.
        """
        assert isinstance(callback, TrainingCallback), "callback must be an instance of TrainingCallback"
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
            raise AssertionError(
                f"Duplicates of 'TrainingCallback' with 'allow_duplicates' == False are not allowed in "
                f"'TrainingCallbackList'. Found duplicates {duplicates}."
            )

    def check_for_duplicate_callback_names(self):
        """
        Check if there is any duplicate callback names in the TrainingCallbackList.
        """
        seen, duplicates = [], []

        for callback in self.callbacks:
            if callback.name in seen:
                duplicates.append(callback.name)
            else:
                seen.append(callback.name)

        if duplicates:
            raise AssertionError(
                f"Duplicates callback names are not allowed in 'TrainingCallbackList'. Found duplicates {duplicates}."
            )

    def remove(self, callback: TrainingCallback):
        """
        Remove a callback from the list.

        Parameters
        ----------
        callback : Callback
            The callback to remove.
        """
        assert isinstance(callback, TrainingCallback), "callback must be an instance of TrainingCallback"
        self.callbacks.remove(callback)
        self.sort()

    def sort(self):
        """
        Sorts the callbacks using their priority.
        """
        self.callbacks.sort(key=lambda callback: callback.priority, reverse=True)

    def state_dict(self) -> Dict[str, Any]:
        """
        Collates the states of the callbacks in a dictionary. This is called when the checkpoint manager saves the
        state of the trainer. Then those states are saved in the checkpoint file with the name of the callback as the
        key.

        Returns
        -------
        states: Dict[str, Any]
            The state of the callback.
        """
        return {callback.name: callback.state_dict() for callback in self.callbacks}

    def on_fit_start(self, trainer, **kwargs):
        """
        Called when the fit starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_fit_start(trainer, **kwargs)

    def on_fit_end(self, trainer, **kwargs):
        """
        Called when the fit ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_fit_end(trainer, **kwargs)

    def on_epoch_start(self, trainer, **kwargs):
        """
        Called when an epoch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, **kwargs)

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, **kwargs)

    def on_train_start(self, trainer, **kwargs):
        """
        Called when the train phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_train_start(trainer, **kwargs)

    def on_train_end(self, trainer, **kwargs):
        """
        Called when the train phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_train_end(trainer, **kwargs)

    def on_train_batch_start(self, trainer, **kwargs):
        """
        Called when the training batch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_train_batch_start(trainer, **kwargs)

    def on_train_batch_end(self, trainer, **kwargs):
        """
        Called when the training batch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(trainer, **kwargs)

    def on_optimization_start(self, trainer, **kwargs):
        """
        Called when the optimization phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_optimization_start(trainer, **kwargs)

    def on_optimization_end(self, trainer, **kwargs):
        """
        Called when the optimization phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_optimization_end(trainer, **kwargs)

    def on_validation_start(self, trainer, **kwargs):
        """
        Called when the validation phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_validation_start(trainer, **kwargs)

    def on_validation_end(self, trainer, **kwargs):
        """
        Called when the validation phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_validation_end(trainer, **kwargs)

    def on_validation_batch_start(self, trainer, **kwargs):
        """
        Called when the validation batch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_validation_batch_start(trainer, **kwargs)

    def on_validation_batch_end(self, trainer, **kwargs):
        """
        Called when the validation batch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        for callback in self.callbacks:
            callback.on_validation_batch_end(trainer, **kwargs)

    def on_progress_bar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        updated_dict = {}
        for callback in self.callbacks:
            callback_dict = callback.on_progress_bar_update(trainer, **kwargs)
            if callback_dict is None:
                callback_dict = {}
            elif not isinstance(callback_dict, dict):
                callback_dict = {callback.name: callback_dict}
            updated_dict.update(callback_dict)
        return updated_dict
