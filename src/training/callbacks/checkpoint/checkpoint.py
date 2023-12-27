"""
    @file:              checkpoint.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     12/2022
    @Last modification: 08/2023

    @Description:       This file is used to define the 'Checkpoint' callback.
"""

from dataclasses import asdict
from itertools import count
import os
from typing import Optional

import torch

from ..base import Priority, TrainingCallback
from .state import CheckpointState


class Checkpoint(TrainingCallback):
    """
    This class is used to manage and create the checkpoints of a model during the training process.
    """

    instance_counter = count()

    CHECKPOINT_EPOCH_KEY: str = "epoch"

    EPOCHS_FOLDER_NAME: str = "epochs"
    FIGURES_FOLDER_NAME = "figures"
    BEST_CHECKPOINT_NAME = "best_model_checkpoint.pt"

    def __init__(
            self,
            epoch_to_start_save: int = 0,
            name: Optional[str] = None,
            path_to_checkpoint_folder: str = "./checkpoints",
            save_freq: int = 1,
            save_model_state: bool = True,
            save_training_history_state: bool = True,
            save_learning_algorithms_state: bool = False,
            save_training_state: bool = False,
            verbose: bool = False
    ):
        """
        Initializes the checkpoint folder.

        Parameters
        ----------
        epoch_to_start_save : int
            The epoch at which to start saving checkpoints.
        name : Optional[str]
            The name of the callback.
        path_to_checkpoint_folder : str
            Path to the folder to save the checkpoints to.
        save_freq : int
            The frequency at which to save checkpoints. If 'save_freq' <= 0, only save at the end of the training. Note
            that the epoch state is saved at every epoch, but the model, optimizer and training state is only saved at
            the given frequency.
        save_model_state : bool
            Whether to include model state in checkpoint.
        save_learning_algorithms_state : bool
            Whether to include learning algorithms state in checkpoint.
        save_training_history_state : bool
            Whether to include training history state in checkpoint.
        save_training_state : bool
            Whether to include training state in checkpoint.
        verbose : bool
            Whether to print out the trace of the checkpoint.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, save_state=False)

        self.epoch_to_start_save = epoch_to_start_save
        self.path_to_checkpoint_folder = path_to_checkpoint_folder
        self.path_to_training_history = os.path.join(self.path_to_checkpoint_folder, self.EPOCHS_FOLDER_NAME)

        self.save_freq = save_freq
        self.save_training_history_state = save_training_history_state
        self.save_learning_algorithms_state = save_learning_algorithms_state
        self.save_model_state = save_model_state
        self.save_training_state = save_training_state
        self.verbose = verbose

    def __getitem__(self, epoch: int) -> dict:
        """
        Loads the checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch index.

        Returns
        -------
        loaded_checkpoint : dict
            The loaded checkpoint.
        """
        if os.path.exists(self._get_path_to_checkpoint(epoch)):
            checkpoint = torch.load(self._get_path_to_checkpoint(epoch))
        else:
            raise FileNotFoundError("File not found.")

        return checkpoint

    def get_best_checkpoint(self, epoch: int) -> dict:
        """
        Loads the checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch index.

        Returns
        -------
        loaded_checkpoint : dict
            The loaded checkpoint.
        """
        if os.path.exists(self._get_path_to_best_checkpoint()):
            checkpoint = torch.load(self._get_path_to_best_checkpoint())
        else:
            raise FileNotFoundError("File not found.")

        return checkpoint

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'TrainingCallbackList'.

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
        return Priority.LOW_PRIORITY

    def save(self, trainer, best: bool = False):
        """
        Saves the checkpoint if the current epoch is a checkpoint epoch.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        best : bool
            Whether to save the checkpoint as the best model.
        """
        if best:
            self._save_checkpoint(
                epoch=trainer.training_state.best_epoch if trainer.is_using_early_stopping else trainer.epoch_state.idx,
                epoch_state=trainer.epoch_state.state_dict(),
                training_state=trainer.training_state.state_dict(),
                model_state=trainer.model.state_dict(),
                learning_algo=None,
                training_history=trainer.training_history.state_dict(),
                best=True
            )
        else:
            os.makedirs(self.path_to_training_history, exist_ok=True)

            self._save_checkpoint(
                epoch=trainer.epoch_state.idx,
                epoch_state=trainer.epoch_state.state_dict(),
                training_state=trainer.training_state.state_dict() if self.save_training_state else None,
                model_state=trainer.model.state_dict() if self.save_model_state else None,
                learning_algo=trainer.learning_algorithms.state_dict() if self.save_learning_algorithms_state else None,
                training_history=trainer.training_history.state_dict() if self.save_training_history_state else None
            )

    def _get_checkpoint_filename(self, epoch: int):
        """
        Generate the filename for the checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch to generate the filename for.

        Returns
        -------
        checkpoint_filename : str
            The filename for the checkpoint at the given epoch.
        """
        return f"{self.CHECKPOINT_EPOCH_KEY}-{epoch}.pt"

    def _get_path_to_best_checkpoint(self) -> str:
        """
        Gets the path to the best model checkpoint.

        Returns
        -------
        path_to_best_checkpoint : str
            The path to the best model checkpoint.
        """
        return os.path.join(self.path_to_checkpoint_folder, self.BEST_CHECKPOINT_NAME)

    def _get_path_to_checkpoint(self, epoch: int) -> str:
        """
        Generates the path to the file for the checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch to generate the filepath for.

        Returns
        -------
        checkpoint_filepath : str
            The filepath for the checkpoint at the given epoch.
        """
        return os.path.join(self.path_to_training_history, self._get_checkpoint_filename(epoch))

    def _save_checkpoint(
            self,
            epoch: int,
            epoch_state: dict,
            model_state: Optional[dict] = None,
            training_state: Optional[dict] = None,
            learning_algo: Optional[dict] = None,
            training_history: Optional[dict] = None,
            best: bool = False
    ):
        """
        Saves a checkpoint (model, optimizer, trainer) states at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch index.
        epoch_state : dict
            The epoch state.
        model_state : dict
            The state dict of the model.
        training_state : dict
            The training state.
        learning_algo : dict
            The state dict of the learning algorithm.
        training_history : dict
            The training history.
        best : bool
            Whether the current epoch is associated to the best checkpoint.
        """
        os.makedirs(self.path_to_checkpoint_folder, exist_ok=True)

        state = CheckpointState(
            epoch=epoch,
            learning_algorithms_state=learning_algo,
            training_history_state=training_history,
            epoch_state=epoch_state,
            model_state=model_state,
            training_state=training_state
        )

        path = self._get_path_to_best_checkpoint() if best else self._get_path_to_checkpoint(epoch)
        torch.save(asdict(state), path)

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends. The checkpoint is saved if the current constraints are met.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        if self.epoch_to_start_save > trainer.epoch_state.idx:
            return
        elif self.save_freq > 0 and trainer.epoch_state.idx % self.save_freq == 0:
            self.save(trainer)

    def on_fit_end(self, trainer, **kwargs):
        """
        Called when the training is finished. Saves the current checkpoint if the current epoch is lower than
        the number of epochs i.e. there is new stuff to save.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        self.save(trainer, True)

        path_to_save = os.path.join(self.path_to_checkpoint_folder, self.FIGURES_FOLDER_NAME)
        os.makedirs(path_to_save, exist_ok=True)

        trainer.training_history.plot(
            path_to_save=path_to_save,
            show=False
        )
