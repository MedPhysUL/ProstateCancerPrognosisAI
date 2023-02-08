"""
    @file:              states.py
    @Author:            Maxence Larose

    @Creation Date:     02/2023
    @Last modification: 02/2023

    @Description:       This file is used to define the multiple useful states storing information about the training
                        process.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict
from typing_extensions import TypeAlias

from monai.data import DataLoader
import numpy as np

from src.training.callbacks.training_history import MeasurementsContainer, MeasurementsType, TrainingHistory
from src.data.datasets.prostate_cancer import FeaturesType, TargetsType
from src.tasks.task_list import TaskList
from src.utils.transforms import to_numpy


MeasurementType: TypeAlias = Dict[str, Dict[str, float]]


@dataclass
class State:
    """
    Abstract state.
    """

    def state_dict(self):
        return asdict(self)


@dataclass
class BatchState(State):
    """
    This class is used to store the current batch state. It is extremely useful for the callbacks to access the
    current batch state and to modify the training process.

    Elements
    --------
    idx : int
        The index of the current batch.
    multi_task_losses_with_regularization : MeasurementType
        The multi-task losses of the current batch, including the regularization term (penalty). The keys are the names
        of the learning algorithm used, the values are dictionaries whose keys are the names of the losses while its
        values are losses measured in the current batch.
    multi_task_losses_without_regularization : MeasurementType
        The multi-task losses of the current batch, excluding the regularization term (penalty). The keys are the names
        of the learning algorithm used, the values are dictionaries whose keys are the names of the losses while its
        values are losses measured in the current batch.
    single_task_losses : MeasurementType
        The single task losses of the current batch. The keys are the names of the tasks, the values are dictionaries
        whose keys are the names of the losses while its values are losses measured in the current batch.
    x : FeaturesType
        The current input batch.
    y : TargetsType
        The current target batch.
    pred : TargetsType
        The current prediction.
    """
    idx: int = None
    multi_task_losses_with_regularization: MeasurementType = field(default_factory=dict)
    multi_task_losses_without_regularization: MeasurementType = field(default_factory=dict)
    single_task_losses: MeasurementType = field(default_factory=dict)
    x: FeaturesType = None
    y: TargetsType = field(default_factory=dict)
    pred: TargetsType = field(default_factory=dict)


@dataclass
class BatchesState(State):
    """
    This class is used to store the current batch state. It is extremely useful for the callbacks to access the
    current batches state and to modify the training process.

    Elements
    --------
    multi_task_losses_with_regularization : MeasurementTypes
        The multi-task losses of the current batches, including the regularization term (penalty). The keys are the
        names of the learning algorithm used, the values are dictionaries whose keys are the names of the losses while
        its values are losses or list of losses measured in the current batch.
    multi_task_losses_without_regularization : MeasurementTypes
        The multi-task losses of the current batches, excluding the regularization term (penalty). The keys are the
        names of the learning algorithm used, the values are dictionaries whose keys are the names of the losses while
        its values are losses or list of losses measured in the current batch.
    single_task_losses : MeasurementTypes
        The single task losses of the current batches. The keys are the names of the tasks, the values are dictionaries
        whose keys are the names of the losses while its values are losses or list of losses measured in the current
        batch.
    """
    multi_task_losses_with_regularization: MeasurementsType = field(default_factory=dict)
    multi_task_losses_without_regularization: MeasurementsType = field(default_factory=dict)
    single_task_losses: MeasurementsType = field(default_factory=dict)

    def init(self, batch_state: BatchState):
        """
        Initializes the current 'BatchesState' using a single 'BatchState'. This method actually defines the keys of
        the measurement dictionaries using the keys present in the given batch state.

        Parameters
        ----------
        batch_state : BatchState
            A batch state.
        """
        for k, v in vars(batch_state).items():
            if k in vars(self):
                vars(self)[k] = {}
                for name, measurements in v.items():
                    vars(self)[k][name] = {}
                    for key, value in measurements.items():
                        vars(self)[k][name][key] = [to_numpy(value)]
            else:
                pass

    def append(self, batch_state: BatchState):
        """
        Appends a batch state to the batches state container.

        Parameters
        ----------
        batch_state : BatchState
            Batch state.
        """
        for k, v in vars(batch_state).items():
            if k in vars(self):
                for name, measurements in v.items():
                    for key, value in measurements.items():
                        vars(self)[k][name][key].append(to_numpy(value))
            else:
                pass

    def mean(self):
        """
        Calculates the average value of all measurements and updates the measurements' dictionaries of the current batch
        state with these values. Be careful, the values of the measurements will become floats instead of lists.
        """
        for k, v in vars(self).items():
            for name, measurements in v.items():
                for key, value in measurements.items():
                    vars(self)[k][name][key] = np.nanmean(value)
            else:
                pass


@dataclass
class EpochState(State):

    SUFFIX_WITH_REGULARIZATION = "('regularization'=True)"
    SUFFIX_WITHOUT_REGULARIZATION = "('regularization'=False)"

    """
    This class is used to store the current epoch state. It is extremely useful for the callbacks to access the
    current epoch state and to modify the training process.

    Elements
    --------
    idx : int
        The index of the current epoch.
    train : MeasurementsContainer
        Training set measurements in the current epoch.
    valid : MeasurementsContainer
        Validation set measurements in the current epoch.
    """
    idx: int = None
    train: MeasurementsContainer = field(default_factory=MeasurementsContainer)
    valid: MeasurementsContainer = field(default_factory=MeasurementsContainer)

    def set_to_last_epoch_of_history(self, history: TrainingHistory):
        """
        Sets the epoch state to the last epoch state of the given training history.

        Parameters
        ----------
        history : TrainingHistory
            Training history.
        """
        last_epoch_hist = history[-1]
        self.train, self.valid = last_epoch_hist.train, last_epoch_hist.valid

    def set_losses_from_batches_state(self, batches_state: BatchesState, training: bool):
        """
        Sets the losses' measurements using the given batches state.

        Parameters
        ----------
        batches_state : BatchesState
            A batches state.
        training : bool
            Whether the model is currently being trained.
        """
        batches_state.mean()

        multi_task_losses = {}
        for algo_name, v in batches_state.multi_task_losses_without_regularization.items():
            multi_task_losses[algo_name] = {}
            for loss_name, loss_value in v.items():
                multi_task_losses[algo_name][f"{loss_name}{self.SUFFIX_WITHOUT_REGULARIZATION}"] = loss_value

                if batches_state.multi_task_losses_with_regularization:
                    loss_with_reg = batches_state.multi_task_losses_with_regularization[algo_name][loss_name]
                    multi_task_losses[algo_name][f"{loss_name}{self.SUFFIX_WITH_REGULARIZATION}"] = loss_with_reg

        if training:
            self.train.multi_task_losses = multi_task_losses
            self.train.single_task_losses = batches_state.single_task_losses
        else:
            self.valid.multi_task_losses = multi_task_losses
            self.valid.single_task_losses = batches_state.single_task_losses


@dataclass
class TrainingState(State):
    """
    This class is used to store the current training state. It is extremely useful for the callbacks to access the
    current training state and to modify the training process.

    Elements
    --------
    info : Dict[str, Any]
        Any additional information. This is useful to communicate between callbacks.
    max_epochs : int
        Maximum number of epochs for training.
    stop_training_flag : bool
        Whether the training should be stopped.
    tasks: TaskList
        List of tasks that the model must learn to perform.
    train_dataloader : Dataloader
        Training set data loader.
    valid_dataloader : Dataloader
        Validation set data loader.
    """
    info: Dict[str, Any] = field(default_factory=dict)
    max_epochs: int = None
    stop_training_flag: bool = False
    tasks: TaskList = None
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None

    def state_dict(self):
        unserializable = ["info", "tasks", "train_dataloader", "valid_dataloader"]
        state_dict = super().state_dict()
        return {k: v for k, v in state_dict if k not in unserializable}
