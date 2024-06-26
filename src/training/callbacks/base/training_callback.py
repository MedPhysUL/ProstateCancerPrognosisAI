"""
    @file:              training_callback.py
    @Author:            Maxence Larose

    @Creation Date:     10/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the TrainingCallback abstract class. A lot of the logic behind the
                        following code is borrowed from PyTorch Lightning
                        (https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) and
                        NeuroTorch (https://github.com/NeuroTorch/NeuroTorch).
"""

from abc import ABC, abstractmethod


class TrainingCallback(ABC):
    """
    Abstract base class used to build training callbacks. A callback is a self-contained program that can be used to
    monitor or alter the training process. The advantage of callbacks is that it decouples a lot of non-essential logic
    that does not need to live in the trainer or the model.

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
        - Valid batch: One forward pass through a single batch. Usually, the batch size used in validation is 1.
        - Validation: Passing through the entire validation dataset (in batches).

    Callback methods are called in the following order:
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
        - `on_fit_end`
    """

    UNSERIALIZABLE_ATTRIBUTES = ["trainer"]

    def __init__(
            self,
            name: str,
            save_state: bool = True
    ):
        """
        Initialize name and state of the training callback.

        Parameters
        ----------
        name : str
            The name of the callback.
        save_state : bool
            Whether to save the state of the callback in the checkpoint file. Default is True.
        """
        self.name = name
        self.save_state = save_state
        self.trainer = None

    @property
    @abstractmethod
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'CallbackList'.

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

    def state_dict(self) -> dict:
        """
        Get the state of the callback. This is called when the checkpoint manager saves the state of the trainer. Then
        this state is saved in the checkpoint file with the name of the callback as the key.

        Returns
        -------
        state : dict
            The state of the callback.
        """
        if self.save_state:
            return {k: v for k, v in vars(self).items() if k not in self.UNSERIALIZABLE_ATTRIBUTES}

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
