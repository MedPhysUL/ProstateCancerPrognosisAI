"""
    @file:              callback.py
    @Author:            Maxence Larose

    @Creation Date:     10/2022
    @Last modification: 12/2022

    @Description:       This file is used to define the Callback abstract class. A lot of the logic behind the
                        following code is borrowed from PyTorch Lightning
                        (https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) and
                        NeuroTorch (https://github.com/NeuroTorch/NeuroTorch).
"""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional


class Priority(IntEnum):
    LOW_PRIORITY = 0
    MEDIUM_PRIORITY = 50
    HIGH_PRIORITY = 100


class Callback(ABC):
    """
    Abstract base class used to build callbacks. A callback is a self-contained program that can be used to monitor or
    alter the training process. The advantage of callbacks is that it decouples a lot of non-essential logic that does
    not need to live in the trainer or the model.

    Nomenclature:
        - Backward pass: The process of propagating the network error (loss) from the output layer to the input layer,
            i.e. updating the weights of the model using gradient descent algorithms ("learning").
        - Batch: A group of samples.
        - Epoch: Passing through the entire training and validation dataset (in batches).
        - Fit: The process of fitting a model to the given dataset.
        - Forward pass: The process of calculating the values of the output layer from the input layer.
        - Optimization: The process of calculating the training loss on a batch and executing a backward pass.
        - Train: Passing through the entire training dataset (in batches).
        - Train batch: One forward and backward pass through a single batch. Usually called an 'iteration'.
        - Trial: The process of evaluating an objective function with a given set of hyperparameters.
        - Tuning: The process of selecting the optimal set of hyperparameters.
        - Valid batch: One forward pass through a single batch. Usually, the batch size used in validation is 1.
        - Validation: Passing through the entire validation dataset (in batches).

    Callback methods are called in the following order:
        - `on_tuning_start`
            * Executes n_trials times:
            - `on_trial_start`
            - `on_fit_start`
            - `load_checkpoint_state`
            * Executes n_epochs times:
                - `on_epoch_start`
                - `on_train_start`
                * Executes n_batches times:
                    - `on_train_batch_start`
                    - `on_optimization_start`
                    - `on_optimization_end`
                    - `on_train_batch_end`
                - `on_train_end`
                - `on_validation_start`
                * Executes validation_set_size times:
                    - `on_validation_batch_start`
                    - `on_validation_batch_end`
                - `on_validation_end`
                - `on_epoch_end`
            - `on_fit_end`
            - `on_trial_end`
        - `on_tuning_end`
    """

    instance_counter = 0

    UNSERIALIZABLE_ATTRIBUTES = ["trainer", "tuner"]

    def __init__(
            self,
            name: str,
            save_state: bool = True,
            load_state: Optional[bool] = None
    ):
        """
        Initialize name and state of the callback.

        Parameters
        ----------
        name : str
            The name of the callback.
        save_state :
            Whether to save the state of the callback in the checkpoint file. Default is True.
        load_state:
            Whether to load from the checkpoint file. Default is equal to save_state.
        """
        self.instance_id = self.instance_counter
        self.name = name if name is not None else f"{self.__class__.__name__}<{self.instance_id}>"
        self.__class__.instance_counter += 1

        self.save_state = save_state
        self.load_state = load_state if load_state is not None else save_state
        self.trainer = None
        self.tuner = None

    @abstractmethod
    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        raise NotImplementedError

    @abstractmethod
    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'CallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
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

    def on_trial_start(self, tuner, **kwargs):
        """
        Called when the trial starts.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        pass

    def on_trial_end(self, tuner, **kwargs):
        """
        Called when the trial ends.

        Parameters
        ----------
        tuner : Tuner
            The tuner.
        """
        pass

    def on_fit_start(self, trainer, **kwargs):
        """
        Called when the fit starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        self.trainer = trainer

    def on_fit_end(self, trainer, **kwargs):
        """
        Called when the fit ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        """
        Loads the state of the callback from a dictionary.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        checkpoint : dict
            The dictionary containing all the states of the trainer.
        """
        if self.load_state and checkpoint is not None:
            state = checkpoint.get(self.name, None)
            if state is not None:
                self.__dict__.update(state)

    def get_checkpoint_state(self, trainer, **kwargs) -> object:
        """
        Get the state of the callback. This is called when the checkpoint manager saves the state of the trainer. Then
        this state is saved in the checkpoint file with the name of the callback as the key.

        Parameters
        ----------
        trainer : Trainer
            The trainer.

        Returns
        -------
        state : object
            The state of the callback.
        """
        if self.save_state:
            return {k: v for k, v in self.__dict__.items() if k not in self.UNSERIALIZABLE_ATTRIBUTES}

    def on_epoch_start(self, trainer, **kwargs):
        """
        Called when an epoch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_train_start(self, trainer, **kwargs):
        """
        Called when the train phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_train_end(self, trainer, **kwargs):
        """
        Called when the train phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_train_batch_start(self, trainer, **kwargs):
        """
        Called when the training batch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_train_batch_end(self, trainer, **kwargs):
        """
        Called when the training batch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_optimization_start(self, trainer, **kwargs):
        """
        Called when the optimization phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_optimization_end(self, trainer, **kwargs):
        """
        Called when the optimization phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_validation_start(self, trainer, **kwargs):
        """
        Called when the validation phase starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_validation_end(self, trainer, **kwargs):
        """
        Called when the validation phase ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_validation_batch_start(self, trainer, **kwargs):
        """
        Called when the validation batch starts.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_validation_batch_end(self, trainer, **kwargs):
        """
        Called when the validation batch ends.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        pass

    def on_progress_bar_update(self, trainer, **kwargs) -> dict:
        """
        Called when the progress bar is updated.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        return {}

    def __del__(self):
        """
        Delete instance.
        """
        self.__class__.instance_counter -= 1
