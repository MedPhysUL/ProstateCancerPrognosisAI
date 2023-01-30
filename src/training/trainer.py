from typing import Generator, List, Optional, Union

from monai.data import DataLoader
from torch import device as torch_device
from torch import cuda, no_grad, Tensor
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm

from src.callbacks.callback import Callback
from src.callbacks.callback_list import CallbackList
from src.callbacks.model_checkpoint import CheckpointLoadingMode, ModelCheckpoint
from src.callbacks.training_history import TrainingHistory
from src.callbacks.learning_algorithm import LearningAlgorithm
from src.data.datasets.prostate_cancer_dataset import FeaturesType, ProstateCancerDataset
from src.models.base.base_model import BaseModel
from src.training.states import BatchState, BatchesState, EpochState, TrainingState
from src.utils.transforms import ToTensor


class Trainer:
    def __init__(
            self,
            model: BaseModel,
            callbacks: Optional[Union[Callback, CallbackList, List[Callback]]] = None,
            device: Optional[torch_device] = None,
            verbose: bool = True,
            **kwargs
    ):
        """
        Constructor for Trainer.
        """
        assert model.is_built, "Model must be built before training"
        self.model = model
        self.callbacks = self._get_initialized_callbacks(callbacks)
        self.sort_callbacks()
        self.device = device if device else model.device
        self.verbose = verbose

        self.batch_state = BatchState()
        self.batches_state = BatchesState()
        self.epoch_state = EpochState()
        self.training_state = TrainingState()

        self.x_transform = kwargs.get("x_transform", ToTensor())
        self.y_transform = kwargs.get("y_transform", ToTensor())

        self._checkpoint_loading_mode = None
        self._force_overwrite = None

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
        training_histories = list(filter(lambda x: isinstance(x, TrainingHistory), self.callbacks))
        assert len(training_histories) == 1, "There should be one and only one `TrainingHistory` callback."

        return training_histories[0]

    @property
    def learning_algorithms(self) -> List[LearningAlgorithm]:
        learning_algorithms = list(filter(lambda x: isinstance(x, LearningAlgorithm), self.callbacks))
        assert learning_algorithms, "There should be at least one `LearningAlgorithm` callback."

        return learning_algorithms

    @property
    def model_checkpoint(self) -> Optional[ModelCheckpoint]:
        model_checkpoints = list(filter(lambda x: isinstance(x, ModelCheckpoint), self.callbacks))

        if model_checkpoints:
            assert len(model_checkpoints) == 1, "There should be a single `ModelCheckpoint` callback, if there is one."
            return model_checkpoints[0]
        else:
            return None

    @staticmethod
    def _get_initialized_callbacks(
            callbacks: Optional[Union[Callback, CallbackList, List[Callback]]]
    ) -> CallbackList:
        if callbacks is None:
            callbacks = []
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        if not any([isinstance(callback, TrainingHistory) for callback in callbacks]):
            callbacks.append(TrainingHistory())
        return CallbackList(callbacks)

    def _checks_if_training_can_be_stopped(self):
        if not self.training_state.stop_training_flag:
            self.training_state.stop_training_flag = all(
                learning_algorithm.stopped for learning_algorithm in self.learning_algorithms
            )

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
                self.model.load_checkpoint_state(checkpoint)

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

        self.training_state.n_epochs = n_epochs
        self.training_state.train_dataloader = train_dataloader
        self.training_state.valid_dataloader = valid_dataloader
        self.training_state.tasks = dataset.tasks

        self.sort_callbacks()
        self.callbacks.on_fit_start(self)
        self.load_state()
        if self.epoch_state.idx is None:
            self.epoch_state.idx = 0
        if len(self.training_history) > 0:
            self.epoch_state.set_to_last_epoch_of_history(self.training_history)

        progress_bar = tqdm(
            initial=self.epoch_state.idx,
            total=self.training_state.n_epochs,
            desc=kwargs.get("desc", "Training"),
            disable=not self.verbose,
            position=p_bar_position,
            unit="epoch",
            leave=p_bar_leave
        )
        for epoch_idx in self._epochs_generator(progress_bar):
            self.epoch_state.idx = epoch_idx
            self.callbacks.on_epoch_start(self)
            train_dataloader = self.training_state.train_dataloader
            valid_dataloader = self.training_state.valid_dataloader

            self._exec_epoch(train_dataloader, valid_dataloader)

            if exec_metrics_on_train or valid_dataloader:
                self.model.fix_thresholds_to_optimal_values(dataset)
            if exec_metrics_on_train:
                self._exec_metrics(dataset=dataset, training=True)
            if valid_dataloader is not None:
                self._exec_metrics(dataset=dataset, training=False)

            self.callbacks.on_epoch_end(self)

            postfix = self.epoch_state.as_dict()
            postfix.update(self.callbacks.on_progress_bar_update(self))
            progress_bar.set_postfix(postfix)

            self._checks_if_training_can_be_stopped()
            if self.training_state.stop_training_flag:
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
        while self.epoch_state.idx < self.training_state.n_epochs:
            yield self.epoch_state.idx

            self.epoch_state.idx += 1
            progress_bar.total = self.training_state.n_epochs
            progress_bar.update()

    def _exec_epoch(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: Optional[DataLoader] = None
    ):
        with no_grad():
            cuda.empty_cache()

        self._exec_training(train_dataloader)

        if valid_dataloader is not None:
            with no_grad():
                self._exec_validation(valid_dataloader)

        with no_grad():
            cuda.empty_cache()

    def _exec_metrics(
            self,
            dataset: ProstateCancerDataset,
            training: bool
    ):
        if training:
            scores = self.model.score_on_dataset(dataset=dataset, mask=dataset.train_mask)
            self.epoch_state.train_single_task_metrics = scores
        else:
            scores = self.model.score_on_dataset(dataset=dataset, mask=dataset.valid_mask)
            self.epoch_state.valid_single_task_metrics = scores

    def _exec_training(
            self,
            dataloader: DataLoader,
    ):
        self.model.train()
        self.callbacks.on_train_start(self)
        self._exec_batches(dataloader)
        self.epoch_state.set_losses_from_batches_state(self.batches_state, self.model.training)
        self.callbacks.on_train_end(self)

    def _exec_validation(
            self,
            dataloader: DataLoader,
    ):
        self.model.eval()
        self.callbacks.on_validation_start(self)
        self._exec_batches(dataloader)
        self.epoch_state.set_losses_from_batches_state(self.batches_state, self.model.training)
        self.callbacks.on_validation_end(self)

    def _exec_batches(
            self,
            dataloader: DataLoader
    ):
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            self.batch_state.idx = batch_idx
            self._exec_batch(x_batch, y_batch)

            if batch_idx == 0:
                self.batches_state.init(self.batch_state)
            else:
                self.batches_state.append(self.batch_state)

    def _exec_batch(
            self,
            x_batch,
            y_batch,
    ):
        x_batch = self.x_transform(self._batch_to_device(x_batch))
        y_batch = self.y_transform(self._batch_to_device(y_batch))

        self.batch_state.x = x_batch
        self.batch_state.y = y_batch

        if self.model.training:
            self.callbacks.on_train_batch_start(self)
            pred_batch = self.model(x_batch)
        else:
            self.callbacks.on_validation_batch_start(self)
            with no_grad():
                pred_batch = self.model(x_batch)
        self.batch_state.pred = pred_batch

        if self.model.training:
            self.callbacks.on_optimization_start(self)
            self.callbacks.on_optimization_end(self)
            self.callbacks.on_train_batch_end(self)
        else:
            self.callbacks.on_validation_batch_end(self)

    def _batch_to_device(
            self,
            batch: Union[dict, FeaturesType, Tensor]
    ) -> Union[dict, FeaturesType, Tensor]:
        """
        Send batch to device.

        Parameters
        ----------
        batch : Union[dict, FeaturesType, Tensor]
            Batch data
        """
        if isinstance(batch, FeaturesType):
            image_to_device = {k: self._batch_to_device(v) for k, v in batch.image.items()}
            table_to_device = {k: self._batch_to_device(v) for k, v in batch.table.items()}
            return FeaturesType(image=image_to_device, table=table_to_device)
        if isinstance(batch, dict):
            return {k: self._batch_to_device(v) for k, v in batch.items()}
        if isinstance(batch, Tensor):
            return batch.to(self.device)
        return batch
