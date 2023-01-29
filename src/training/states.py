from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from monai.data import DataLoader
import numpy as np
from torch import Tensor

from src.callbacks.training_history import MeasurementHistoryType, TrainingHistory
from src.data.datasets.prostate_cancer_dataset import FeaturesModel
from src.utils.tasks import Task
from src.utils.transforms import to_numpy


MeasurementType = Dict[str, Dict[str, float]]
MeasurementTypes = Union[MeasurementType, MeasurementHistoryType]


@dataclass
class BaseState:

    def __getstate__(self):
        return {k: v for k, v in vars(self).items()}


@dataclass
class BatchState(BaseState):
    """
    This class is used to store the current batch state.

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
    x : DataModel.x
        The current input batch.
    y : DataModel.y
        The current target batch.
    pred : DataModel.y
        The current prediction.
    """
    idx: int = None
    multi_task_losses_with_regularization: MeasurementType = field(default_factory=dict)
    multi_task_losses_without_regularization: MeasurementType = field(default_factory=dict)
    single_task_losses: MeasurementType = field(default_factory=dict)
    x: FeaturesModel = None
    y: Dict[str, Tensor] = field(default_factory=dict)
    pred: Dict[str, Tensor] = field(default_factory=dict)


@dataclass
class BatchesState(BaseState):
    """
    This class is used to store the current batch state.

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
    multi_task_losses_with_regularization: MeasurementTypes = field(default_factory=dict)
    multi_task_losses_without_regularization: MeasurementTypes = field(default_factory=dict)
    single_task_losses: MeasurementTypes = field(default_factory=dict)

    def init(self, batch_state: BatchState):
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
        Append a batch state to the batches state container.

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
        for k, v in vars(self).items():
            for name, measurements in v.items():
                for key, value in measurements.items():
                    vars(self)[k][name][key] = np.nanmean(value)
            else:
                pass


@dataclass
class EpochState(BaseState):

    _SUFFIX_WITH_REGULARIZATION = "('regularization'=True)"
    _SUFFIX_WITHOUT_REGULARIZATION = "('regularization'=False)"

    """
    This class is used to store the current epoch state.

    Elements
    --------
    idx : int
        The index of the current epoch.
    train_multi_task_losses : MeasurementType
        The multi-task losses of the training set in the current epoch.
    valid_multi_task_losses : MeasurementType
        The multi-task losses of the validation set in the current epoch.
    train_single_task_losses : MeasurementType
        The single task losses of the training set in the current epoch. Keys are task names.
    valid_single_task_losses : MeasurementType
        The single task losses of the validation set in the current epoch. Keys are task names.
    train_single_task_metrics : MeasurementType
        The single task metrics of the training set in the current epoch.
    valid_single_task_metrics : MeasurementType
        The single task metrics of the validation set in the current epoch.
    """
    idx: int = None
    train_multi_task_losses: MeasurementType = field(default_factory=dict)
    valid_multi_task_losses: MeasurementType = field(default_factory=dict)
    train_single_task_losses: MeasurementType = field(default_factory=dict)
    valid_single_task_losses: MeasurementType = field(default_factory=dict)
    train_single_task_metrics: MeasurementType = field(default_factory=dict)
    valid_single_task_metrics: MeasurementType = field(default_factory=dict)

    def set_to_last_epoch_of_history(self, history: TrainingHistory):
        last_epoch_hist = history[-1]
        train_measures, valid_measures = last_epoch_hist[TrainingHistory.TRAIN], last_epoch_hist[TrainingHistory.VALID]

        self.train_multi_task_losses = train_measures[TrainingHistory.MULTI_TASK_LOSSES]
        self.train_single_task_losses = train_measures[TrainingHistory.SINGLE_TASK_LOSSES]
        self.train_single_task_metrics = train_measures[TrainingHistory.SINGLE_TASK_METRICS]

        self.valid_multi_task_losses = valid_measures[TrainingHistory.MULTI_TASK_LOSSES]
        self.valid_single_task_losses = valid_measures[TrainingHistory.SINGLE_TASK_LOSSES]
        self.valid_single_task_metrics = valid_measures[TrainingHistory.SINGLE_TASK_METRICS]

    def set_losses_from_batches_state(self, batches_state: BatchesState, training: bool):
        batches_state.mean()

        multi_task_losses = {}
        for algo_name, v in batches_state.multi_task_losses_without_regularization.items():
            multi_task_losses[algo_name] = {}
            for loss_name, loss_value in v.items():
                multi_task_losses[algo_name][f"{loss_name}{self._SUFFIX_WITHOUT_REGULARIZATION}"] = loss_value

                if batches_state.multi_task_losses_with_regularization:
                    loss_with_reg = batches_state.multi_task_losses_with_regularization[algo_name][loss_name]
                    multi_task_losses[algo_name][f"{loss_name}{self._SUFFIX_WITH_REGULARIZATION}"] = loss_with_reg

        if training:
            self.train_multi_task_losses = multi_task_losses
            self.train_single_task_losses = batches_state.single_task_losses
        else:
            self.valid_multi_task_losses = multi_task_losses
            self.valid_single_task_losses = batches_state.single_task_losses

    def as_dict(self):
        epoch_state_as_dict = {
            TrainingHistory.TRAIN: {
                TrainingHistory.MULTI_TASK_LOSSES: self.train_multi_task_losses,
                TrainingHistory.SINGLE_TASK_LOSSES: self.train_single_task_losses,
                TrainingHistory.SINGLE_TASK_METRICS: self.train_single_task_metrics
            },
            TrainingHistory.VALID: {
                TrainingHistory.MULTI_TASK_LOSSES: self.valid_multi_task_losses,
                TrainingHistory.SINGLE_TASK_LOSSES: self.valid_single_task_losses,
                TrainingHistory.SINGLE_TASK_METRICS: self.valid_single_task_metrics
            }
        }

        return epoch_state_as_dict


@dataclass
class TrainingState(BaseState):
    """
    This class is used to store the current training state. It is extremely useful for the callbacks to access the
    current training state and to modify the training process.

    Elements
    --------
    info : Dict[str, Any]
        Any additional information. This is useful to communicate between callbacks.
    n_epochs : int
        Total number of epochs.
    stop_training_flag : bool
        Whether the training should be stopped.
    tasks: List[Task]
        List of tasks that the model must learn to perform.
    train_dataloader : Dataloader
        Training set data loader.
    valid_dataloader : Dataloader
        Validation set data loader.
    """
    info: Dict[str, Any] = field(default_factory=dict)
    n_epochs: int = None
    stop_training_flag: bool = False
    tasks: List[Task] = field(default_factory=list)
    train_dataloader: DataLoader = None
    valid_dataloader: DataLoader = None

    def __getstate__(self):
        unserializable = ["info", "tasks", "train_dataloader", "valid_dataloader"]
        return {k: v for k, v in vars(self).items() if k not in unserializable}
