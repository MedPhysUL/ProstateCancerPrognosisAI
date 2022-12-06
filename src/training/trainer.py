
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union

from monai.data import DataLoader
import numpy as np
from torch import cuda, device, no_grad, Tensor
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm

from src.callbacks.callback import Callback
from src.callbacks.callback_list import CallbackList
from src.callbacks.model_checkpoint import CheckpointLoadingMode, ModelCheckpoint
from src.callbacks.training_history import MeasureCategory, TrainingHistory
from src.data.datasets.prostate_cancer_dataset import FeaturesModel, ProstateCancerDataset
from src.data.processing.tools import MaskType
# TODO : Move MaskType elsewhere and (Transform Masktype in Enum or MeasureCategory in a classic class).
from src.models.base.base_model import BaseModel
from src.utils.transforms import to_numpy, ToTensor


@dataclass
class TrainingState:
    """
    This class is used to store the current training state. It is extremely useful for the callbacks to access the
    current training state and to modify the training process.

    Elements
    --------
    batch : int
        The current batch.
    batch_loss : float
        The current loss.
    epoch : int
        The current epoch.
    epoch_losses_and_metrics : Dict[str, Dict[str, Dict[str, float]]]
        The current epoch losses and metrics, i.e the current training set losses and metrics and validation set
        losses and metrics.
    info : Dict[str, Any]
        Any additional information. This is useful to communicate between callbacks.
    n_epochs : int
        Total number of epochs.
    objects : Dict[str, Any]
        Any additional objects. This is useful to manage objects between callbacks. Note: In general, the
        train_dataloader, valid_dataloader and tasks objetcs should be stored here.
    pred_batch : DataModel.y
        The current prediction.
    stop_training_flag : bool
        Whether the training should be stopped.
    x_batch : DataModel.x
        The current input batch.
    y_batch : DataModel.y
        The current target batch.
    """
    batch: Optional[int] = None
    batch_loss: Optional[float] = None
    epoch: Optional[int] = None
    epoch_losses_and_metrics: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)
    n_epochs: Optional[int] = None
    objects: Dict[str, Any] = field(default_factory=dict)
    pred_batch: Optional[Dict[str, Tensor]] = None
    stop_training_flag: bool = False
    x_batch: Optional[FeaturesModel] = None
    y_batch: Optional[Dict[str, Tensor]] = None

    def __getstate__(self):
        unserializable = ["info", "objects"]
        d = {k: v for k, v in vars(self).items() if k not in unserializable}
        return d

    def update(self, **kwargs):
        self_dict = vars(self)
        assert all(k in self_dict for k in kwargs)
        self_dict.update(kwargs)

    @property
    def train_losses(self) -> Dict[str, float]:
        return self.epoch_losses_and_metrics[MaskType.TRAIN][MeasureCategory.LOSSES.value]

    @property
    def train_metrics(self) -> Dict[str, float]:
        return self.epoch_losses_and_metrics[MaskType.TRAIN][MeasureCategory.METRICS.value]

    @property
    def valid_losses(self) -> Dict[str, float]:
        return self.epoch_losses_and_metrics[MaskType.VALID][MeasureCategory.LOSSES.value]

    @property
    def valid_metrics(self) -> Dict[str, float]:
        return self.epoch_losses_and_metrics[MaskType.VALID][MeasureCategory.METRICS.value]


class Trainer:
    def __init__(
            self,
            model: BaseModel,
            callbacks: Optional[Union[Callback, CallbackList, List[Callback]]] = None,
            device: Optional[device] = None,
            verbose: bool = True,
            **kwargs
    ):
        """
        Constructor for Trainer.
        """
        assert model.is_built, "Model must be built before training"
        self.model = model
        self.callbacks = self._set_default_callbacks(callbacks)
        self.sort_callbacks()
        self.device = self._set_default_device(device)
        self.verbose = verbose
        self.state = TrainingState()

        self.x_transform = kwargs.get("x_transform", ToTensor())
        self.y_transform = kwargs.get("y_transform", ToTensor())

        self._checkpoint_loading_mode = None
        self._force_overwrite = None

    @property
    def model(self):
        """
        Alias for the model.

        :return: The :attr:`model` attribute.
        """
        return self.model

    @model.setter
    def model(self, value):
        """
        Alias for the model.

        :param value: The new value for the :attr:`model` attribute.
        :return: None
        """
        self.model = value

    @property
    def checkpoint_loading_mode(self):
        return self._checkpoint_loading_mode

    @checkpoint_loading_mode.setter
    def checkpoint_loading_mode(self, value: CheckpointLoadingMode):
        self._checkpoint_loading_mode = value

    @property
    def force_overwrite(self):
        return self._force_overwrite

    @property
    def training_history(self) -> TrainingHistory:
        training_histories = list(filter(lambda x: isinstance(x, TrainingHistory), self.callbacks))[0]
        assert len(training_histories) == 1, "There should be one and only one 'TrainingHistory' callback."

        return training_histories[0]

    @property
    def model_checkpoint(self) -> Optional[ModelCheckpoint]:
        model_checkpoints = list(filter(lambda x: isinstance(x, ModelCheckpoint), self.callbacks))

        if model_checkpoints:
            assert len(model_checkpoints) == 1, "There should be a single 'ModelCheckpoint' callback, if there is one."
            return model_checkpoints[0]
        else:
            return None

    def _set_default_device(self, device: Optional[device]) -> device:
        if device is None:
            return self.model.device
        return device

    @staticmethod
    def _set_default_callbacks(
            callbacks: Optional[Union[Callback, CallbackList, List[Callback]]]
    ) -> CallbackList:
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        if not any([isinstance(callback, TrainingHistory) for callback in callbacks]):
            callbacks.append(TrainingHistory())
        return CallbackList(callbacks)

    def update_state(self, **kwargs):
        self.state.update(**kwargs)

    def update_objects_state(self, **kwargs):
        self.update_state(objects={**self.state.objects, **kwargs})

    def update_epoch_losses_and_metrics_state(self, **kwargs):
        self.update_state(epoch_losses_and_metrics={**self.state.epoch_losses_and_metrics, **kwargs})

    def update_info_state(self, **kwargs):
        self.update_state(info={**self.state.info, **kwargs})

    def update_train_losses(self, train_losses: Dict[str, float]):
        self.state.epoch_losses_and_metrics[MaskType.TRAIN][MeasureCategory.LOSSES.value] = train_losses

    def update_train_metrics(self, train_metrics: Dict[str, float]):
        self.state.epoch_losses_and_metrics[MaskType.TRAIN][MeasureCategory.METRICS.value] = train_metrics

    def update_valid_losses(self, valid_losses: Dict[str, float]):
        self.state.epoch_losses_and_metrics[MaskType.VALID][MeasureCategory.LOSSES.value] = valid_losses

    def update_valid_metrics(self, valid_metrics: Dict[str, float]):
        self.state.epoch_losses_and_metrics[MaskType.VALID][MeasureCategory.METRICS.value] = valid_metrics

    def sort_callbacks(self):
        """
        Sort the callbacks by their priority. The higher the priority, the earlier the callback is called. In general,
        the callbacks will be sorted in the following order:
            1. TrainingHistory callbacks;
            2. Others callbacks;
            3. CheckpointManager callbacks.
        """
        self.callbacks.sort()

    def load_state(self):
        """
        Load the state of the trainer from the checkpoint.
        """
        if self.model_checkpoint:
            checkpoint = self.model_checkpoint.current_checkpoint
            if checkpoint:
                self.callbacks.load_checkpoint_state(self, checkpoint)

    @staticmethod
    def _create_train_dataloader(
            dataset: ProstateCancerDataset,
            batch_size: int
    ) -> DataLoader:
        """
        Creates the objects needed for the training.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloaders.
        batch_size : int
            Size of the batches in the train loader.

        Returns
        -------
        train_loader : DataLoader
            Train loader.
        """
        # Creation of training loader
        train_size = len(dataset.train_mask)
        batch_size = min(train_size, batch_size) if batch_size is not None else train_size

        train_data = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(dataset.train_mask),
            drop_last=(train_size % batch_size) == 1,
            collate_fn=None
        )

        return train_data

    @staticmethod
    def _create_valid_dataloader(
            dataset: ProstateCancerDataset,
            batch_size: int = 1
    ) -> DataLoader:
        """
        Creates the objects needed for validation during the training process.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloader.
        batch_size : int
            Size of the batches in the valid loader.

        Returns
        -------
        validation_loader : DataLoader
            Validation loader.
        """
        # We create the valid dataloader (if valid size != 0)
        valid_size, valid_data = len(dataset.valid_mask), None

        if valid_size != 0:

            # We check if a valid batch size was provided
            valid_batch_size = min(valid_size, batch_size)

            # We create the valid loader
            valid_data = DataLoader(
                dataset,
                batch_size=valid_batch_size,
                sampler=SubsetRandomSampler(dataset.valid_mask),
                collate_fn=None
            )

        return valid_data

    def train(
            self,
            dataset: ProstateCancerDataset,
            n_epochs: int = 100,
            exec_metrics_on_train: bool = True,
            batch_size: int = 8,
            checkpoint_loading_mode: Optional[CheckpointLoadingMode] = None,
            force_overwrite: bool = False,
            p_bar_position: Optional[int] = None,
            p_bar_leave: Optional[bool] = None,
            **kwargs
    ) -> TrainingHistory:
        """
        Train the model.

        :return: The training history.
        """
        self._checkpoint_loading_mode = checkpoint_loading_mode
        self._force_overwrite = force_overwrite

        train_dataloader = self._create_train_dataloader(dataset=dataset, batch_size=batch_size)
        valid_dataloader = self._create_valid_dataloader(dataset=dataset)

        self.update_state(
            n_epochs=n_epochs,
            objects={
                **self.state.objects,
                **{"train_dataloader": train_dataloader, "valid_dataloader": valid_dataloader, "tasks": dataset.tasks}
            }
        )

        self.sort_callbacks()
        self.callbacks.on_fit_start(self)

        self.load_state()
        if self.state.epoch is None:
            self.update_state(epoch=0)
        if len(self.training_history) > 0:
            self.update_epoch_losses_and_metrics_state(**self.training_history[-1])

        progress_bar = tqdm(
            initial=self.state.epoch,
            total=self.state.n_epochs,
            desc=kwargs.get("desc", "Training"),
            disable=not self.verbose,
            position=p_bar_position,
            unit="epoch",
            leave=p_bar_leave
        )
        for epoch in self._epochs_generator(progress_bar):
            self.update_state(epoch=epoch)
            self.callbacks.on_epoch_start(self)
            train_dataloader = self.state.objects["train_dataloader"]
            valid_dataloader = self.state.objects["valid_dataloader"]

            epoch_loss = self._exec_epoch(train_dataloader, valid_dataloader)

            if exec_metrics_on_train:
                itr_train_metrics = self._exec_metrics(train_dataloader, prefix="train")
            else:
                itr_train_metrics = {}

            if valid_dataloader is not None:
                itr_val_metrics = self._exec_metrics(valid_dataloader, prefix="val")
            else:
                itr_val_metrics = {}

            self.callbacks.on_epoch_end(self)

            postfix = {f"{k}": f"{v:.5e}" for k, v in self.state.itr_metrics.items()}
            postfix.update(self.callbacks.on_progress_bar_update(self))
            progress_bar.set_postfix(postfix)
            if self.state.stop_training_flag:
                progress_bar.set_postfix(dict(**{"stop_flag": "True"}, **postfix))
                break
        self.callbacks.on_fit_end(self)
        progress_bar.close()

        return self.training_history

    def _epochs_generator(self, progress_bar: tqdm) -> Generator:
        """
        Generator that yields the current epoch and updates the state and the progress bar.

        :return: The current iteration.
        """
        while self.state.epoch < self.state.n_epochs:
            yield self.state.epoch

            self.update_state(epoch=self.state.epoch + 1)
            progress_bar.total = self.state.n_epochs
            progress_bar.update()

    def _exec_epoch(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, float]:
        with no_grad():
            cuda.empty_cache()
        losses = {}

        train_loss = self._exec_training(train_dataloader)

        losses["train_loss"] = train_loss

        if valid_dataloader is not None:
            with no_grad():
                val_loss = self._exec_validation(valid_dataloader)
                losses["val_loss"] = val_loss

        with no_grad():
            cuda.empty_cache()

        return losses

    def _exec_metrics(self, dataloader: DataLoader, prefix: str) -> Dict:
        metrics_dict = {}
        for metric in self.metrics:
            m_out = metric(dataloader)
            if isinstance(m_out, dict):
                metrics_dict.update({f"{prefix}_{k}": v for k, v in m_out.items()})
            else:
                metric_name = str(metric)
                if hasattr(metric, "name"):
                    metric_name = metric.name
                elif hasattr(metric, "__name__"):
                    metric_name = metric.__name__
                metrics_dict[f"{prefix}_{metric_name}"] = m_out
        return metrics_dict

    def _exec_training(
            self,
            dataloader: DataLoader,
    ) -> float:
        self.model.train()
        self.callbacks.on_train_start(self)

        train_loss = self._exec_batches(dataloader)

        self.update_state_(train_loss=train_loss)
        self.callbacks.on_train_end(self)

        return train_loss

    def _exec_validation(
            self,
            dataloader: DataLoader,
    ) -> float:
        self.model.eval()
        self.callbacks.on_validation_start(self)

        val_loss = self._exec_batches(dataloader)

        self.update_state_(val_loss=val_loss)
        self.callbacks.on_validation_end(self)

        return val_loss

    def _exec_batches(
            self,
            dataloader: DataLoader
    ):
        batch_losses = []
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            self.update_state(batch=batch_idx)
            batch_losses.append(to_numpy(self._exec_batch(x_batch, y_batch)))

        return np.nanmean(batch_losses)

    def _exec_batch(
            self,
            x_batch,
            y_batch,
    ):
        x_batch = self.x_transform(self._batch_to_device(x_batch))
        y_batch = self.y_transform(self._batch_to_device(y_batch))
        self.update_state(x_batch=x_batch, y_batch=y_batch)

        if self.model.training:
            self.callbacks.on_train_batch_start(self)
            pred_batch = self.model(x_batch)
        else:
            self.callbacks.on_validation_batch_start(self)
            with no_grad():
                pred_batch = self.model(x_batch)

        self.update_state(pred_batch=pred_batch)

        if self.model.training:
            self.callbacks.on_optimization_start(self)
            self.callbacks.on_optimization_end(self)
            self.callbacks.on_train_batch_end(self)
        else:
            self.callbacks.on_validation_batch_end(self)

        batch_loss = self.state.batch_loss

        return batch_loss

    def _batch_to_device(
            self,
            batch: Union[dict, FeaturesModel, Tensor]
    ) -> Union[dict, FeaturesModel, Tensor]:
        """
        Send batch to device.

        Parameters
        ----------
        batch : Union[dict, FeaturesModel, Tensor]
            Batch data
        """
        if isinstance(batch, FeaturesModel):
            image_to_device = {k: self._batch_to_device(v) for k, v in batch.image.items()}
            table_to_device = {k: self._batch_to_device(v) for k, v in batch.table.items()}
            return FeaturesModel(image=image_to_device, table=table_to_device)
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        return batch
