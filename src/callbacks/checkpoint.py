"""
    @file:              checkpoint.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the 'Checkpoint' callback.
"""

from dataclasses import asdict, dataclass
from itertools import count
import json
import os
from typing import Any, Dict, Mapping, Optional, Union

import torch

from src.callbacks.callback import Callback, Priority


@dataclass
class CheckpointState:
    """
    A data class which is used to hold important information about the checkpoint at the current training epoch.
    """
    callbacks_state: dict
    epoch: int
    epoch_state: dict
    model_state: Optional[dict]
    training_state: dict


class Checkpoint(Callback):
    """
    This class is used to manage and create the checkpoints of a model during the training process.
    """

    instance_counter = count()

    CHECKPOINT_EPOCH_KEY: str = "epoch"
    CHECKPOINT_EPOCHS_KEY: str = "epochs"
    CHECKPOINT_METADATA_KEY: str = 'metadata'

    TRAINING_HISTORY_FIGURE_NAME = "training_history.png"

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

        os.makedirs(path_to_checkpoint_folder, exist_ok=True)

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
        with open(self.path_to_checkpoints_metadata, "r+") as file:
            info = json.load(file)
        filename = self._get_save_name_from_checkpoints(info, epoch)
        checkpoint = torch.load(f"{self.path_to_checkpoint_folder}/{filename}")

        return checkpoint

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
    def path_to_checkpoints_metadata(self) -> str:
        """
        Gets the path to the checkpoints metadata file.

        Returns
        -------
        metadata_path : str
            The path to the checkpoints metadata file.
        """
        return os.path.join(self.path_to_checkpoint_folder, f"{self.CHECKPOINT_METADATA_KEY}.json")

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.LOW_PRIORITY.value

    def save(self, trainer) -> bool:
        """
        Saves the checkpoint if the current epoch is a checkpoint epoch.

        Parameters
        ----------
        trainer : Trainer
            The trainer.

        Returns
        -------
        saved : bool
            Whether the checkpoint was saved.
        """
        self._save_checkpoint(
            epoch=trainer.epoch_state.idx,
            epoch_state=trainer.epoch_state.state_dict(),
            training_state=trainer.training_state.state_dict(),
            model_state=trainer.model.state_dict(),
            callbacks_state=trainer.callbacks.state_dict()
        )

        trainer.training_history.plot(
            save_path=os.path.join(self.path_to_checkpoint_folder, self.TRAINING_HISTORY_FIGURE_NAME),
            show=False
        )

        return True

    def _create_new_checkpoint_metadata(self, epoch: int) -> dict:
        """
        Creates a new checkpoint's metadata.

        Parameters
        ----------
        epoch : int
            The epoch of the checkpoint.

        Returns
        -------
        metadata : dict
            The new checkpoint's metadata.
        """
        save_name = self._get_checkpoint_filename(epoch)
        return {self.CHECKPOINT_EPOCHS_KEY: {str(epoch): save_name}}

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

    def _get_save_name_from_checkpoints(
            self,
            checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
            epoch: int
    ) -> str:
        """
        Gets the save name from the checkpoint's metadata given the load checkpoint mode.

        Parameters
        ----------
        checkpoints_meta : Dict[str, Union[str, Dict[Any, str]]]
            The checkpoint's metadata.
        epoch : int
            The epoch to get the checkpoint at.

        Returns
        -------
        save_name : str
            The save name.
        """
        if str(epoch) in checkpoints_meta[self.CHECKPOINT_EPOCHS_KEY].keys():
            return checkpoints_meta[self.CHECKPOINT_EPOCHS_KEY][str(epoch)]
        else:
            raise ValueError(f"Invalid epoch index {epoch}.")

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
        save_name = self._get_checkpoint_filename(epoch)
        path = os.path.join(self.path_to_checkpoint_folder, save_name)

        state = CheckpointState(
            callbacks_state=callbacks_state,
            epoch=epoch,
            epoch_state=epoch_state,
            model_state=model_state if self.save_model_state else None,
            training_state=training_state
        )

        torch.save(asdict(state), path)
        self._save_checkpoints_metadata(self._create_new_checkpoint_metadata(epoch))
        return path

    def _save_checkpoints_metadata(self, new_info: dict):
        """
        Saves the new checkpoints' metadata.

        Parameters
        ----------
        new_info : dict
            The new checkpoints' metadata.
        """
        info = {}
        if os.path.exists(self.path_to_checkpoints_metadata):
            with open(self.path_to_checkpoints_metadata, "r+") as file:
                info = json.load(file)
        self._update_dictionary_recursively(info, new_info)
        os.makedirs(self.path_to_checkpoint_folder, exist_ok=True)
        with open(self.path_to_checkpoints_metadata, "w+") as file:
            json.dump(info, file, indent=4)

    def _update_dictionary_recursively(self, dict_to_update: dict, updater: Mapping) -> dict:
        """
        Update dict recursively.

        Parameters
        ----------
        dict_to_update : dict
            Mapping item that wil be updated.
        updater : Mapping
            Mapping item updater.

        Returns
        -------
        updated_dict : dict
            Updated dict.
        """
        for k, v in updater.items():
            if isinstance(v, Mapping):
                dict_to_update[k] = self._update_dictionary_recursively(dict_to_update.get(k, {}), v)
            else:
                dict_to_update[k] = v
        return dict_to_update

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
        elif trainer.state.epoch >= trainer.training_state.n_epochs - 1:
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
        if trainer.epoch_state.idx < trainer.training_state.n_epochs:
            self.save(trainer)
