"""
    @file:              checkpoint.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     12/2022
    @Last modification: 02/2023

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

    BEST_MODEL_KEY: str = "best"
    CHECKPOINT_EPOCH_KEY: str = "epoch"
    CHECKPOINT_EPOCHS_KEY: str = "epochs"

    TRAINING_HISTORY_FIGURES_FOLDER_NAME = "training_history_figures"

    def __init__(
            self,
            epoch_to_start_save: int = 0,
            name: Optional[str] = None,
            path_to_checkpoint_folder: str = "./checkpoints",
            save_freq: int = 1,
            save_model_state: bool = True,
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
            The frequency at which to save checkpoints. If 'save_freq' <= 0, only save at the end of the training.
        save_model_state : bool
            Whether to include model state in checkpoint.
        verbose : bool
            Whether to print out the trace of the checkpoint.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, save_state=False)

        self.epoch_to_start_save = epoch_to_start_save
        self.path_to_checkpoint_folder = path_to_checkpoint_folder
        self.save_freq = save_freq
        self.save_model_state = save_model_state
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
        elif os.path.exists(self._get_path_to_checkpoint(epoch, True)):
            checkpoint = torch.load(self._get_path_to_checkpoint(epoch, True))
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

    def save(self, trainer):
        """
        Saves the checkpoint if the current epoch is a checkpoint epoch.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        self._save_checkpoint(
            epoch=trainer.epoch_state.idx,
            epoch_state=trainer.epoch_state.state_dict(),
            training_state=trainer.training_state.state_dict(),
            model_state=trainer.model.state_dict(),
            callbacks_state=trainer.callbacks.state_dict()
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
        return f"{self.CHECKPOINT_EPOCH_KEY}-{epoch}.pth"

    def _get_best_checkpoint_filename(self, epoch: int):
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
        return f"{self.CHECKPOINT_EPOCH_KEY}-{epoch}({self.BEST_MODEL_KEY}).pth"

    def _get_path_to_checkpoint(self, epoch: int, best: bool = False) -> str:
        """
        Generates the path to the file for the checkpoint at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch to generate the filepath for.
        best : bool
            Whether the current epoch is associated to the best checkpoint.

        Returns
        -------
        checkpoint_filepath : str
            The filepath for the checkpoint at the given epoch.
        """
        if best:
            return os.path.join(self.path_to_checkpoint_folder, self._get_best_checkpoint_filename(epoch))
        else:
            return os.path.join(self.path_to_checkpoint_folder, self._get_checkpoint_filename(epoch))

    def _save_checkpoint(
            self,
            callbacks_state: dict,
            epoch: int,
            epoch_state: dict,
            model_state: dict,
            training_state: dict,
    ) -> str:
        """
        Saves a checkpoint (model, optimizer, trainer) states at the given epoch.

        Parameters
        ----------
        callbacks_state : dict
            The state dict of all callbacks.
        epoch : int
            The epoch index.
        epoch_state : dict
            The epoch state.
        model_state : dict
            The state dict of the model.
        training_state : dict
            The training state.

        Returns
        -------
        path : str
            The path to the saved checkpoint.
        """
        os.makedirs(self.path_to_checkpoint_folder, exist_ok=True)
        path = self._get_path_to_checkpoint(epoch)

        state = CheckpointState(
            callbacks_state=callbacks_state,
            epoch=epoch,
            epoch_state=epoch_state,
            model_state=model_state if self.save_model_state else None,
            training_state=training_state
        )

        torch.save(asdict(state), path)
        return path

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
        elif trainer.epoch_state.idx >= trainer.training_state.n_epochs - 1:
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
        if trainer.is_using_early_stopping:
            best_epoch_idx = trainer.training_state.best_epoch
            path_to_best_checkpoint = self._get_path_to_checkpoint(best_epoch_idx)
            if os.path.exists(path_to_best_checkpoint):
                os.rename(
                    path_to_best_checkpoint,
                    self._get_path_to_checkpoint(best_epoch_idx, True)
                )

        path_to_save = os.path.join(self.path_to_checkpoint_folder, self.TRAINING_HISTORY_FIGURES_FOLDER_NAME)
        os.makedirs(path_to_save, exist_ok=True)

        trainer.training_history.plot(
            path_to_save=path_to_save,
            show=False
        )
