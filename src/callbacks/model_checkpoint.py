"""
    @file:              checkpoint.py
    @Author:            Maxence Larose, Mehdi Mitiche, Nicolas Raymond

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       This file is used to define the 'Checkpoint' callback.
"""

from enum import Enum
from itertools import count
import json
import os
import shutil
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import torch

from src.callbacks.callback import Callback, Priority


class BestModelMode(Enum):
    """
    Enum for the different ways to define the best model.

    Elements
    --------
    MULTITASK_LOSS : str
        The model with the smallest multi-task loss on the validation set is considered the 'best' model.
    """
    MULTITASK_LOSS = "MultiTaskLoss"
    # TODO : Implement a multi-task score metric.


class CheckpointLoadingMode(Enum):
    """
    Enum for the different modes of loading a checkpoint.

    Elements
    --------
    BEST_EPOCH : str
        The epoch with the best metric is loaded.
    LAST_EPOCH : str
        The last epoch is loaded.
    """
    BEST_EPOCH = "best"
    LAST_EPOCH = "last"


class ModelCheckpoint(Callback):
    """
    This class is used to manage and create the checkpoints of a model.
    """

    instance_counter = count()

    CHECKPOINT_METADATA_KEY: str = 'metadata'
    CHECKPOINT_BEST_KEY: str = "best"
    CHECKPOINT_EPOCHS_KEY: str = "epochs"
    CHECKPOINT_EPOCH_KEY: str = "epoch"
    CHECKPOINT_EPOCH_LOSSES_AND_METRICS_KEY: str = 'epoch_losses_and_metrics'
    CHECKPOINT_OPTIMIZER_STATE_KEY: str = "optimizer_state"
    CHECKPOINT_MODEL_STATE_KEY: str = "model_state"
    CHECKPOINT_TRAINING_HISTORY_KEY: str = "training_history"

    def __init__(
            self,
            path_to_checkpoint_folder: str = "./checkpoints",
            save_freq: int = 1,
            save_best_only: bool = False,
            epoch_to_start_save: int = 0,
            verbose: bool = False,
            name: Optional[str] = None,
            **kwargs
    ):
        """
        Initializes the checkpoint folder.

        Parameters
        ----------
        path_to_checkpoint_folder : str
            Path to the folder to save the checkpoints to.
        save_freq : int
            The frequency at which to save checkpoints. If 'save_freq' <= 0, only save at the end of the training.
        save_best_only : bool
            Whether to only save the best checkpoint. If True, 'save_freq' == -1.
        epoch_to_start_save : int
            The epoch at which to start saving checkpoints.
        verbose : bool
            Whether to print out the trace of the checkpoint.
        name : Optional[str]
            The name of the callback.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name is not None else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, save_state=False, **kwargs)

        os.makedirs(path_to_checkpoint_folder, exist_ok=True)
        self.path_to_checkpoint_folder = path_to_checkpoint_folder
        self.verbose = verbose

        self.save_freq = save_freq
        self.save_best_only = save_best_only
        if self.save_best_only:
            self.save_freq = -1
        self.epoch_to_start_save = epoch_to_start_save
        self.current_best_metric = np.inf
        self.current_checkpoint = None

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

    @staticmethod
    def _replace_trainer_history(trainer, new_history: dict):
        """
        Replaces the training history in the trainer with the given history.

        Parameters
        ----------
        trainer : Trainer
            Trainer object.
        new_history : dict
            The new history dictionary.
        """
        trainer.training_history.load_checkpoint_state(trainer, new_history)
        trainer.sort_callbacks()

    @staticmethod
    def get_save_name_from_checkpoints(
            checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
            checkpoint_loading_mode: CheckpointLoadingMode = CheckpointLoadingMode.BEST_EPOCH
    ) -> str:
        """
        Gets the save name from the checkpoint's metadata given the load checkpoint mode.

        Parameters
        ----------
        checkpoints_meta : Dict[str, Union[str, Dict[Any, str]]]
            The checkpoint's metadata.
        checkpoint_loading_mode : CheckpointLoadingMode
            The checkpoint loading mode.

        Returns
        -------
        save_name : str
            The save name.
        """
        if checkpoint_loading_mode == checkpoint_loading_mode.BEST_EPOCH:
            if ModelCheckpoint.CHECKPOINT_BEST_KEY in checkpoints_meta:
                return checkpoints_meta[ModelCheckpoint.CHECKPOINT_BEST_KEY]
            else:
                raise FileNotFoundError(f"No best checkpoint found in checkpoints_meta. Please use a different "
                                        f"'checkpoint_loading_mode'.")
        elif checkpoint_loading_mode == checkpoint_loading_mode.LAST_EPOCH:
            epoch_dict = checkpoints_meta[ModelCheckpoint.CHECKPOINT_EPOCHS_KEY]
            last_epoch: int = max([int(e) for e in epoch_dict])
            return checkpoints_meta[ModelCheckpoint.CHECKPOINT_EPOCHS_KEY][str(last_epoch)]
        else:
            raise ValueError("Invalid 'checkpoint_loading_mode'. ")

    @property
    def path_to_checkpoints_metadata(self) -> str:
        """
        Gets the path to the checkpoints metadata file.

        Returns
        -------
        metadata_path : str
            The path to the checkpoints metadata file.
        """
        return os.path.join(self.path_to_checkpoint_folder, f"{ModelCheckpoint.CHECKPOINT_METADATA_KEY}.json")

    @staticmethod
    def get_checkpoint_filename(epoch: int):
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
        return f"{ModelCheckpoint.CHECKPOINT_EPOCH_KEY}-{epoch}.pth"

    def _create_new_checkpoint_metadata(self, epoch: int, best: bool = False) -> dict:
        """
        Creates a new checkpoint's metadata.

        Parameters
        ----------
        epoch : int
            The epoch of the checkpoint.
        best : bool
            Whether the checkpoint is currently the best.

        Returns
        -------
        metadata : dict
            The new checkpoint's metadata.
        """
        save_name = self.get_checkpoint_filename(epoch)
        new_info = {ModelCheckpoint.CHECKPOINT_EPOCHS_KEY: {str(epoch): save_name}}
        if best:
            new_info[ModelCheckpoint.CHECKPOINT_BEST_KEY] = save_name
        return new_info

    def save_checkpoint(
            self,
            epoch: int,
            epoch_losses_and_metrics: Dict[str, float],
            best: bool = False,
            model_state: Optional[dict] = None,
            optimizer_state: Optional[dict] = None,
            training_history: Optional[Any] = None,
            **other_states,
    ) -> str:
        """
        Saves a checkpoint (model, optimizer, trainer) states at the given epoch.

        Parameters
        ----------
        epoch : int
            The epoch number.
        epoch_losses_and_metrics : Dict[str, Dict[str, Dict[str, float]]]
            Train losses and metrics AND valid losses and metrics at the current epoch.
        best : bool
            Whether this is the best epoch so far.
        model_state : Optional[dict]
            The state dict of the model.
        optimizer_state: Optional[dict]
            The state dict of the optimizer.
        training_history : Optional[Any]
            The training history object.

        Returns
        -------
        path : str
            The path to the saved checkpoint.
        """
        os.makedirs(self.path_to_checkpoint_folder, exist_ok=True)
        save_name = self.get_checkpoint_filename(epoch)
        path = os.path.join(self.path_to_checkpoint_folder, save_name)

        basic_states = {
            ModelCheckpoint.CHECKPOINT_EPOCH_KEY: epoch,
            ModelCheckpoint.CHECKPOINT_MODEL_STATE_KEY: model_state,
            ModelCheckpoint.CHECKPOINT_OPTIMIZER_STATE_KEY: optimizer_state,
            ModelCheckpoint.CHECKPOINT_EPOCH_LOSSES_AND_METRICS_KEY: epoch_losses_and_metrics,
            ModelCheckpoint.CHECKPOINT_TRAINING_HISTORY_KEY: training_history
        }
        assert all(key not in other_states for key in basic_states.keys()), (
            f"Other states cannot have the same keys as the basic states ({list(basic_states.keys())})."
        )
        states = {**basic_states, **other_states}
        torch.save(states, path)
        self.save_checkpoints_metadata(self._create_new_checkpoint_metadata(epoch, best))
        return path

    def load_checkpoint(
            self,
            checkpoint_loading_mode: CheckpointLoadingMode = CheckpointLoadingMode.BEST_EPOCH
    ) -> dict:
        """
        Loads the checkpoint at the given 'checkpoint_loading_mode'.

        Parameters
        ----------
        checkpoint_loading_mode : CheckpointLoadingMode
            The 'checkpoint_loading_mode' to use.

        Returns
        -------
        loaded_checkpoint : dict
            The loaded checkpoint.
        """
        with open(self.path_to_checkpoints_metadata, "r+") as file:
            info = json.load(file)
        filename = self.get_save_name_from_checkpoints(info, checkpoint_loading_mode)
        checkpoint = torch.load(f"{self.path_to_checkpoint_folder}/{filename}")

        return checkpoint

    def save_checkpoints_metadata(self, new_info: dict):
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

    def on_fit_start(self, trainer, **kwargs):
        """
        Load the checkpoint base on the 'checkpoint_loading_mode' of the trainer and update the 'state' of the trainer.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        super().on_fit_start(trainer)
        if trainer.checkpoint_loading_mode is None:
            trainer.checkpoint_loading_mode = CheckpointLoadingMode.LAST_EPOCH

        epoch = 0
        checkpoint = self.current_checkpoint
        if trainer.force_overwrite:
            if os.path.exists(self.path_to_checkpoints_metadata):
                shutil.rmtree(self.path_to_checkpoint_folder)
        else:
            try:
                checkpoint = self.load_checkpoint(trainer.checkpoint_loading_mode)
                trainer.model.load_state(checkpoint[ModelCheckpoint.CHECKPOINT_MODEL_STATE_KEY], strict=True)
                if trainer.optimizer is not None:
                    trainer.optimizer.load_state(checkpoint[ModelCheckpoint.CHECKPOINT_OPTIMIZER_STATE_KEY])
                epoch = int(checkpoint[ModelCheckpoint.CHECKPOINT_EPOCH_KEY]) + 1

            except FileNotFoundError as e:
                if self.verbose:
                    print("No such checkpoint. Fit from beginning.")
            finally:
                self.current_checkpoint = checkpoint

        trainer.update_state(epoch=epoch)
        self.current_best_metric = np.nanmin(
            trainer.training_history.validation_set_history.losses[trainer.criterion.name]
        )

    def save_on(self, trainer) -> bool:
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
        current_valid_losses = trainer.state.valid_losses

        other_states = trainer.callbacks.get_checkpoint_state(trainer)
        is_best = self._check_is_best(trainer)
        if is_best:
            self.current_best_metric = current_valid_losses[trainer.criterion.name]

        self.save_checkpoint(
            epoch=trainer.state.epoch,
            epoch_losses_and_metrics=trainer.state.epoch_losses_and_metrics,
            best=is_best,
            state_dict=trainer.model.state_dict(),
            optimizer_state_dict=trainer.optimizer.state_dict() if trainer.optimizer else None,
            training_history=trainer.training_history.get_checkpoint_state(trainer),
            **other_states
        )
        if trainer.training_history:
            trainer.training_history.plot(
                save_path=os.path.join(self.path_to_checkpoint_folder, "training_history.png"),
                show=False
            )
        return True

    def on_epoch_end(self, trainer, **kwargs):
        """
        Called when an epoch ends. The checkpoint is saved if the current constraints are met.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        if self.epoch_to_start_save > trainer.state.epoch:
            return
        if self.save_best_only:
            if self._check_is_best(trainer):
                self.save_on(trainer)
        elif self.save_freq > 0 and trainer.state.epoch % self.save_freq == 0:
            self.save_on(trainer)
        if trainer.state.epoch >= trainer.state.n_epochs - 1:
            self.save_on(trainer)

    def _check_is_best(self, trainer) -> Optional[bool]:
        """
        Whether the current epoch is considered the best.

        Parameters
        ----------
        trainer

        Returns
        -------
        is_best : bool
            Whether the current epoch is considered the best.
        """
        if trainer.state.valid_losses is None:
            return None

        return trainer.state.valid_losses[trainer.criterion.name] > self.current_best_metric

    def on_fit_end(self, trainer, **kwargs):
        """
        Called when the training is finished. Saves the current checkpoint if the current epoch is lower than
        the number of epochs i.e. there is new stuff to save.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        if trainer.state.epoch < trainer.state.n_epochs:
            self.save_on(trainer)
