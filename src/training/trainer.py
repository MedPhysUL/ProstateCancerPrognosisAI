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

from os import makedirs
from shutil import rmtree
from typing import Generator, List, NamedTuple, Optional, Sequence, Union
from uuid import uuid1

import numpy as np
from monai.data import DataLoader
from monai.utils import set_determinism
from torch import device as torch_device
from torch import cuda, no_grad, random
from torch.nn import Identity
from torch.utils.data import SubsetRandomSampler
from tqdm.auto import tqdm

from .callbacks import Checkpoint, LearningAlgorithm, TrainingHistory
from .callbacks.base import TrainingCallback
from .callbacks.containers import TrainingCallbackList
from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..models.torch.base import TorchModel
from .states import BatchState, BatchesState, EpochState, TrainingState
from ..tools.transforms import batch_to_device, ToTensor


class TrainingResult(NamedTuple):
    """
    Training result.

    Elements
    --------
    model : TorchModel
        Trained model.
    training_history : TrainingHistory
        Training history.
    """
    model: TorchModel
    training_history: TrainingHistory


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
            n_epochs: int = 100,
            seed: Optional[int] = None,
            valid_batch_size: Optional[int] = None,
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
        n_epochs : int
            Maximum number of epochs for training. Default is 100.
        seed : Optional[int]
            Random state used for reproducibility.
        valid_batch_size : Optional[int]
            Size of the batches in the dataset loader. Default is equal to the training loader batch size.
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
        self.n_epochs = n_epochs
        self.model = None
        self.valid_batch_size = valid_batch_size if valid_batch_size else batch_size
        self.verbose = verbose
        self._seed = seed

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

        self._learning_algorithms[-1].is_last = True
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

    @property
    def is_using_early_stopping(self) -> bool:
        """
        Whether any learning algorithm is using early stopping.

        Returns
        -------
        trainer_is_using_early_stopping : bool
            Whether any learning algorithm is using early stopping.
        """
        return any(learn_algo.early_stopper for learn_algo in self.learning_algorithms)

    def _initialize_states(self):
        """
        Initializes all states.
        """
        self.batch_state = BatchState()
        self.batches_state = BatchesState()
        self.epoch_state = EpochState()
        self.training_state = TrainingState(n_epochs=self.n_epochs)

    def _check_if_training_can_be_stopped(self):
        """
        Checks if the training process can be stopped. Training can be stopped if any learning algorithm is stopped.
        """
        if not self.training_state.stop_training_flag:
            self.training_state.stop_training_flag = any(
                learning_algorithm.stopped for learning_algorithm in self.learning_algorithms
            )

    def _delete_temporary_folder(self):
        """
        Deletes the temporary folder.
        """
        if self.is_using_early_stopping:
            rmtree(self.training_state.path_to_temporary_folder)

    def _set_temporary_folder(self, path_to_temporary_folder: Optional[str]):
        """
        Sets path to temporary folder and creates the directory on the computer.

        Parameters
        ----------
        path_to_temporary_folder : Optional[str]
            Path to temporary folder.
        """
        if self.is_using_early_stopping:
            if path_to_temporary_folder is None:
                path_to_temporary_folder = f"./EarlyStopTemporaryFolder-{uuid1()}"
            makedirs(path_to_temporary_folder, exist_ok=False)
            self.training_state.path_to_temporary_folder = path_to_temporary_folder

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

        rng_state = random.get_rng_state()
        train_data = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size,
            sampler=SubsetRandomSampler(dataset.train_mask),
            drop_last=(train_size % train_batch_size) < train_batch_size / 2,
            collate_fn=None
        )
        random.set_rng_state(rng_state)

        return train_data

    @staticmethod
    def _create_valid_dataloader(
            dataset: ProstateCancerDataset,
            batch_size: int
    ) -> Optional[DataLoader]:
        """
        Creates the dataloader needed for validation during the training process.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloader.
        batch_size : int
            Size of the batches in the valid loader.

        Returns
        -------
        valid_loader : DataLoader
            Validation loader.
        """
        valid_size, valid_data = len(dataset.valid_mask), None

        if valid_size != 0:
            valid_batch_size = min(valid_size, batch_size)

            rng_state = random.get_rng_state()
            valid_data = DataLoader(
                dataset=dataset,
                batch_size=valid_batch_size,
                sampler=SubsetRandomSampler(dataset.valid_mask),
                drop_last=(valid_size % valid_batch_size) < valid_batch_size / 2,
                collate_fn=None
            )
            random.set_rng_state(rng_state)

        return valid_data

    def train(
            self,
            model: TorchModel,
            dataset: ProstateCancerDataset,
            learning_algorithms: Union[LearningAlgorithm, List[LearningAlgorithm]],
            path_to_temporary_folder: Optional[str] = None,
            p_bar_position: Optional[int] = None,
            p_bar_leave: Optional[bool] = None,
            **kwargs
    ) -> TrainingResult:
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
        path_to_temporary_folder : Optional[str]
            Path to temporary folder used to save the best model during run time. The folder is only created when early
            stopping is used and the folder is deleted at the end of the training.
        p_bar_position : Optional[int]
            The position of the progress bar. See https://tqdm.github.io documentation for more information.
        p_bar_leave :  Optional[bool]
            Whether to leave the progress bar. See https://tqdm.github.io documentation for more information.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        training_result : TrainingResult
            The training result.
        """
        self._set_seed()
        self.device = self.device if self.device else model.device
        self.learning_algorithms = learning_algorithms
        self.model = model

        self._initialize_states()
        self._set_temporary_folder(path_to_temporary_folder)

        train_dataloader = self._create_train_dataloader(dataset=dataset, batch_size=self.batch_size)
        valid_dataloader = self._create_valid_dataloader(dataset=dataset, batch_size=self.valid_batch_size)

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

            self._exec_epoch(self.training_state.train_dataloader, self.training_state.valid_dataloader)

            if self.exec_metrics_on_train or valid_dataloader:
                self.model.fix_thresholds_to_optimal_values(dataset)
            if self.exec_metrics_on_train:
                self._exec_metrics(dataset=dataset, training=True)
            if valid_dataloader is not None:
                self._exec_metrics(dataset=dataset, training=False)

            self.callbacks.on_epoch_end(self)

            postfix = self._get_progress_postfix()
            postfix.update(self.callbacks.on_progress_bar_update(self))
            progress_bar.set_postfix(postfix)

            self._check_if_training_can_be_stopped()
            if self.training_state.stop_training_flag:
                progress_bar.set_postfix(dict(**{"stop_flag": "True"}, **postfix))
                break

        self._exec_post_fit(dataset)
        self.callbacks.on_fit_end(self)
        self._delete_temporary_folder()
        progress_bar.close()

        return TrainingResult(model=self.model, training_history=self.training_history)

    def _get_progress_postfix(self) -> dict:
        """
        Gives the cleaned postfix for the progress bar created from epoch_state.state_dict().

        Returns
        -------
        postfix : dict
            {train_loss: {algorithm: loss}, valid_loss: {algorithm: loss}}.
        """
        postfix = {}
        state_dict = self.epoch_state.state_dict()

        for state_key, state in state_dict.items():
            if isinstance(state, dict):
                postfix_key = state_key + "_loss"
                postfix[postfix_key] = {}
                for algorithm_key, algorithm in state["multi_task_losses"].items():
                    for loss in algorithm.values():
                        postfix[postfix_key][algorithm_key] = np.round_(loss, 3)
        return postfix

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
        dataloader.dataset.enable_augmentations()  # type: ignore
        self.model.train()
        self.callbacks.on_train_start(self)
        self._exec_batches(dataloader)
        self.epoch_state.set_losses_from_batches_state(self.batches_state, self.model.training)
        self.callbacks.on_train_end(self)
        dataloader.dataset.disable_augmentations()  # type: ignore

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
        is usually called an 'iteration'). During validation, this is one forward pass through a single batch.

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
            scores = self.model.compute_score_on_dataset(dataset=dataset, mask=dataset.train_mask)
            self.epoch_state.train.single_task_metrics = scores
        else:
            scores = self.model.compute_score_on_dataset(dataset=dataset, mask=dataset.valid_mask)
            self.epoch_state.valid.single_task_metrics = scores

    def _exec_post_fit(
            self,
            dataset: ProstateCancerDataset
    ):
        """
        Executes some post fit adjustments to the trained model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        """
        self.model.fix_thresholds_to_optimal_values(dataset)
        self.model.fit_breslow_estimators(dataset)

    def _empty_cache(self):
        """
        Empties GPU cache if device's type is "cuda".
        """
        if self.device.type == "cuda":
            with no_grad():
                cuda.empty_cache()

    def _set_seed(self):
        """
        Sets numpy and torch seed.
        """
        if self._seed is not None:
            set_determinism(self._seed)
