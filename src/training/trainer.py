"""
    @file:              trainer.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the Trainer, an object used to train a PyTorch model to perform
                        certain tasks given sample data. Under the hood, the trainer handles all the details of the
                        training loop, like running the training and validation data loaders, calling callbacks at the
                        appropriate times, and more.
"""

from typing import Generator, List, Optional, Sequence, Union

from monai.data import DataLoader
from torch import device as torch_device
from torch import cuda, no_grad
from torch.nn import Identity
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm

from .callbacks.base import TrainingCallback
from .callbacks.containers import TrainingCallbackList
from .callbacks import Checkpoint, LearningAlgorithm, TrainingHistory
from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..models.base.torch_model import TorchModel
from .states import BatchState, BatchesState, EpochState, TrainingState
from ..tools.transforms import batch_to_device, ToTensor


class Trainer:
    """
    This class is used to train a PyTorch model to perform certain tasks given sample data.
    """

    def __init__(
            self,
            batch_size: int = 8,
            checkpoint: Checkpoint = None,
            device: Optional[torch_device] = None,
            exec_metrics_on_train: bool = True,
            max_epochs: int = 100,
            verbose: bool = True,
            **kwargs
    ):
        """
        Constructor for Trainer. Sets the model and callbacks, and initializes batch, batches, epoch, and training
        states to empty states.

        Parameters
        ----------
        batch_size : int
            Size of the batches in the training loader. Default is 8.
        checkpoint : Checkpoint
             Checkpoint used to manage and create the checkpoints of a model during the training process.
        device : Optional[torch_device]
            Device to use for the training process. Default is the device of the model.
        exec_metrics_on_train : bool
            Whether to compute metrics on the training set. This is useful when you want to save time by not computing
            the metrics on the training set. Default is True.
        max_epochs : int
            Maximum number of epochs for training. Default is 100.
        verbose : bool
            Whether to print out the trace of the trainer.
        **kwargs : dict
            x_transform : Module
                Transform to apply to the input data before passing it to the model.
            y_transform : Module
                Transform to apply to the target data before passing it to the model.
        """
        self.batch_size = batch_size
        self.device = device
        self.exec_metrics_on_train = exec_metrics_on_train
        self.max_epochs = max_epochs
        self.model = None
        self.verbose = verbose

        self._checkpoint = checkpoint
        self._learning_algorithms = None
        self._training_history = TrainingHistory()
        self._build_callbacks()

        self.batch_state, self.batches_state, self.epoch_state, self.training_state = None, None, None, None
        self._initialize_states()

        self.x_transform = kwargs.get("x_transform", Identity())
        self.y_transform = kwargs.get("y_transform", Identity())

    @property
    def callbacks(self) -> Optional[TrainingCallbackList]:
        """
        Callbacks to use during the training process.

        Returns
        -------
        callbacks : Optional[TrainingCallbackList]
            Callback list
        """
        return self._callbacks

    def _build_callbacks(self):
        """
        Builds the callbacks attribute as a `CallbackList` and sorts the callbacks in this list.
        """
        callbacks: List[TrainingCallback] = []

        if self.training_history:
            callbacks += [self.training_history]
        if self.learning_algorithms:
            callbacks += self.learning_algorithms
        if self.checkpoint:
            callbacks += [self.checkpoint]

        self._callbacks = TrainingCallbackList(callbacks)
        self._callbacks.sort()

    @property
    def checkpoint(self) -> Optional[Checkpoint]:
        """
        Checkpoint callback.

        Returns
        -------
        checkpoint : Optional[Checkpoint]
            The checkpoint callback, if there is one.
        """
        return self._checkpoint

    @checkpoint.setter
    def checkpoint(self, checkpoint: Optional[Checkpoint]):
        """
        Sets checkpoint callback.

        Parameters
        ----------
        checkpoint : Optional[Checkpoint]
            The checkpoint callback.
        """
        self._checkpoint = checkpoint
        self._build_callbacks()

    @property
    def learning_algorithms(self) -> Optional[Sequence[LearningAlgorithm]]:
        """
        Learning algorithm callbacks.

        Returns
        -------
        learning_algorithms : Optional[Sequence[LearningAlgorithm]]
            The learning algorithm callbacks.
        """
        return self._learning_algorithms

    @learning_algorithms.setter
    def learning_algorithms(self, learning_algorithms: Optional[Union[LearningAlgorithm, Sequence[LearningAlgorithm]]]):
        """
        Sets learning algorithm callbacks.

        Parameters
        ----------
        learning_algorithms : Optional[LearningAlgorithm]
            The learning algorithm callbacks.
        """
        if isinstance(learning_algorithms, Sequence):
            self._learning_algorithms = learning_algorithms
        elif isinstance(learning_algorithms, LearningAlgorithm):
            self._learning_algorithms = [learning_algorithms]
        else:
            raise TypeError(
                "'learning_algorithms' should be of type 'LearningAlgorithm' or 'Sequence[LearningAlgorithm]'"
            )

        self._build_callbacks()

    @property
    def training_history(self) -> TrainingHistory:
        """
        Training history callback.

        Returns
        -------
        training_history : TrainingHistory
            The training history callback.
        """
        return self._training_history

    @training_history.setter
    def training_history(self, training_history: TrainingHistory):
        """
        Sets training history callback.

        Parameters
        ----------
        training_history : TrainingHistory
            The training history callback.
        """
        self._training_history = training_history
        self._build_callbacks()

    def _initialize_states(self):
        """
        Initializes all states.
        """
        self.batch_state = BatchState()
        self.batches_state = BatchesState()
        self.epoch_state = EpochState()
        self.training_state = TrainingState(max_epochs=self.max_epochs)

    def _check_if_training_can_be_stopped(self):
        """
        Checks if the training process can be stopped. Training can be stopped if all learning algorithms are stopped.
        """
        if not self.training_state.stop_training_flag:
            self.training_state.stop_training_flag = all(
                learning_algorithm.stopped for learning_algorithm in self.learning_algorithms
            )

    @staticmethod
    def _create_train_dataloader(
            dataset: ProstateCancerDataset,
            batch_size: int
    ) -> DataLoader:
        """
        Creates the dataloader needed for training.

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
        train_size = len(dataset.train_mask)
        train_batch_size = min(train_size, batch_size) if batch_size is not None else train_size

        train_data = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size,
            sampler=SubsetRandomSampler(dataset.train_mask),
            drop_last=(train_size % train_batch_size) == 1,
            collate_fn=None
        )

        return train_data

    @staticmethod
    def _create_valid_dataloader(
            dataset: ProstateCancerDataset,
            batch_size: int = 1
    ) -> Optional[DataLoader]:
        """
        Creates the dataloader needed for validation during the training process.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloader.
        batch_size : int
            Size of the batches in the valid loader. Default is 1.

        Returns
        -------
        valid_loader : DataLoader
            Validation loader.
        """
        valid_size, valid_data = len(dataset.valid_mask), None

        if valid_size != 0:
            valid_batch_size = min(valid_size, batch_size)

            valid_data = DataLoader(
                dataset=dataset,
                batch_size=valid_batch_size,
                sampler=SubsetRandomSampler(dataset.valid_mask),
                collate_fn=None
            )

        return valid_data

    def train(
            self,
            model: TorchModel,
            dataset: ProstateCancerDataset,
            learning_algorithms: Union[LearningAlgorithm, List[LearningAlgorithm]],
            p_bar_position: Optional[int] = None,
            p_bar_leave: Optional[bool] = None,
            **kwargs
    ) -> TrainingHistory:
        """
        Trains the model.

        Parameters
        ----------
        model : TorchModel
            Model to train.
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloaders.
        learning_algorithms : Union[LearningAlgorithm, List[LearningAlgorithm]]
            The learning algorithm callbacks.
        p_bar_position : Optional[int]
            The position of the progress bar. See https://tqdm.github.io documentation for more information.
        p_bar_leave :  Optional[bool]
            Whether to leave the progress bar. See https://tqdm.github.io documentation for more information.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        training_history : TrainingHistory
            The training history.
        """
        self.device = self.device if self.device else model.device
        self.learning_algorithms = learning_algorithms
        self.model = model

        self._initialize_states()

        train_dataloader = self._create_train_dataloader(dataset=dataset, batch_size=self.batch_size)
        valid_dataloader = self._create_valid_dataloader(dataset=dataset)

        self.training_state.train_dataloader = train_dataloader
        self.training_state.valid_dataloader = valid_dataloader
        self.training_state.tasks = dataset.tasks

        self.callbacks.sort()
        self.callbacks.on_fit_start(self)

        if self.epoch_state.idx is None:
            self.epoch_state.idx = 0
        if not self.training_history.is_empty:
            self.epoch_state.set_to_last_epoch_of_history(self.training_history)

        progress_bar = tqdm(
            initial=self.epoch_state.idx,
            total=self.training_state.max_epochs,
            desc=kwargs.get("desc", "Training"),
            disable=not self.verbose,
            position=p_bar_position,
            unit="epoch",
            leave=p_bar_leave
        )
        for epoch_idx in self._epochs_generator(progress_bar):
            self.epoch_state.idx = epoch_idx
            self.callbacks.on_epoch_start(self)

            self._exec_epoch(self.training_state.train_dataloader, self.training_state.valid_dataloader)

            if self.exec_metrics_on_train or valid_dataloader:
                self.model.fix_thresholds_to_optimal_values(dataset)
            if self.exec_metrics_on_train:
                self._exec_metrics(dataset=dataset, training=True)
            if valid_dataloader is not None:
                self._exec_metrics(dataset=dataset, training=False)

            self.callbacks.on_epoch_end(self)

            postfix = self.epoch_state.state_dict()
            postfix.update(self.callbacks.on_progress_bar_update(self))
            progress_bar.set_postfix(postfix)

            self._check_if_training_can_be_stopped()
            if self.training_state.stop_training_flag:
                progress_bar.set_postfix(dict(**{"stop_flag": "True"}, **postfix))
                break

        self.callbacks.on_fit_end(self)
        progress_bar.close()

        return self.training_history

    def _epochs_generator(self, progress_bar: tqdm) -> Generator:
        """
        Generator that yields the current epoch and updates the current epoch state and the progress bar.

        Parameters
        ----------
        progress_bar : tqdm
            The tqdm progress bar.

        Returns
        -------
        epochs_generator : Generator
            The current epoch.
        """
        while self.epoch_state.idx < self.training_state.max_epochs:
            yield self.epoch_state.idx
            self.epoch_state.idx += 1
            progress_bar.total = self.training_state.max_epochs
            progress_bar.update()

    def _exec_epoch(
            self,
            train_dataloader: DataLoader,
            valid_dataloader: Optional[DataLoader] = None
    ):
        """
        Executes an epoch, i.e. passing through the entire training and validation set (in batches).

        Parameters
        ----------
        train_dataloader : DataLoader
            Training set data loader.
        valid_dataloader : Optional[DataLoader]
            Validation set data loader.
        """
        self._empty_cache()

        self._exec_training(train_dataloader)

        if valid_dataloader is not None:
            with no_grad():
                self._exec_validation(valid_dataloader)

        self._empty_cache()

    def _exec_training(
            self,
            dataloader: DataLoader,
    ):
        """
        Executes training, i.e. passing through the entire training set (in batches).

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader.
        """
        self.model.train()
        self.callbacks.on_train_start(self)
        self._exec_batches(dataloader)
        self.epoch_state.set_losses_from_batches_state(self.batches_state, self.model.training)
        self.callbacks.on_train_end(self)

    def _exec_validation(
            self,
            dataloader: DataLoader,
    ):
        """
        Executes validation, i.e. passing through the entire validation set (in batches).

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader
        """
        self.model.eval()
        self.callbacks.on_validation_start(self)
        self._exec_batches(dataloader)
        self.epoch_state.set_losses_from_batches_state(self.batches_state, self.model.training)
        self.callbacks.on_validation_end(self)

    def _exec_batches(
            self,
            dataloader: DataLoader
    ):
        """
        Executes multiple batches, i.e. one forward (and backward during training) pass through each batch contained in
        the dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader.
        """
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            self.batch_state.idx = batch_idx
            self._exec_batch(x_batch, y_batch)

            if batch_idx == 0:
                self.batches_state.init(self.batch_state)
            else:
                self.batches_state.append(self.batch_state)

    def _exec_batch(
            self,
            x_batch: FeaturesType,
            y_batch: TargetsType,
    ):
        """
        Executes a batch. During training, this is one forward and backward pass through a single batch (this process
        is usually called an 'iteration'). During validation, this is one forward pass through a single batch (usually,
        the batch size used in validation is 1).

        Parameters
        ----------
        x_batch : FeaturesType
            Features batch.
        y_batch : TargetsType
            Targets batch.
        """
        x_batch = self.x_transform(batch_to_device(x_batch, self.device))
        y_batch = self.y_transform(batch_to_device(y_batch, self.device))

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

    def _exec_metrics(
            self,
            dataset: ProstateCancerDataset,
            training: bool
    ):
        """
        Executes metrics, i.e. calculates the score of each metric on the dataset knowing if the model is currently
        being validated or trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        training : bool
            Whether the model is in training or evaluation mode.
        """
        if training:
            scores = self.model.score_on_dataset(dataset=dataset, mask=dataset.train_mask)
            self.epoch_state.train.single_task_metrics = scores
        else:
            scores = self.model.score_on_dataset(dataset=dataset, mask=dataset.valid_mask)
            self.epoch_state.valid.single_task_metrics = scores

    def _empty_cache(self):
        """
        Empties GPU cache if device's type is "cuda".
        """
        if self.device.type == "cuda":
            with no_grad():
                cuda.empty_cache()
