"""
    @file:              custom_torch_base.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file contains an abstract class named TorchCustomModel from which all custom pytorch
                        models implemented for the project must inherit. This class allows to store common function of
                        all pytorch models.
"""


from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

# from dgl import DGLGraph
from torch import tensor, Tensor
from torch.nn import BatchNorm1d, Module
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler

from src.data.datasets.prostate_cancer_dataset import DataModel, ProstateCancerDataset
from src.data.processing.tools import MaskType
# from src.data.processing.gnn_datasets import PetaleKGNNDataset
# from src.models.blocks.mlp_blocks import EntityEmbeddingBlock
from src.training.early_stopper import EarlyStopper, EarlyStopperType
from src.training.optimizer import SAM
from src.utils.multi_task_losses import MultiTaskLoss
from src.visualization.tools import visualize_epoch_progression


class Evaluation(NamedTuple):
    losses: Dict[str, List[float]]
    scores: Dict[str, List[float]]


class TorchCustomModel(Module, ABC):
    """
    Abstract class used to store common attributes and methods of torch models implemented in the project.
    """

    def __init__(
            self,
            criterion: MultiTaskLoss,
            output_size: int,
            path_to_model: str,
            alpha: float = 0,
            beta: float = 0,
            num_cont_col: Optional[int] = None,
            cat_idx: Optional[List[int]] = None,
            cat_sizes: Optional[List[int]] = None,
            cat_emb_sizes: Optional[List[int]] = None,
            additional_input_args: Optional[List[Any]] = None,
            verbose: bool = False
    ) -> None:
        """
        Sets the protected attributes and creates an embedding block if required.

        Parameters
        ----------
        criterion : MultiTaskLoss
            Loss function of our model.
        path_to_model : str
            Path to save model.
        alpha : float
            L1 penalty coefficient.
        beta : float
            L2 penalty coefficient.
        num_cont_col : Optional[int]
            Number of numerical continuous columns in the dataset, cont idx are assumed to be range(num_cont_col).
        cat_idx : Optional[List[int]]
            Idx of categorical columns in the dataset.
        cat_sizes : Optional[List[int]]
            List of integer representing the size of each categorical column.
        cat_emb_sizes : Optional[List[int]]
            List of integer representing the size of each categorical embedding.
        additional_input_args : Optional[List[Any]]
            List of arguments that must be also considered when validating input arguments.
        verbose : bool
            True if we want to print the training progress.
        """
        # We validate input arguments (check if there are continuous or categorical inputs)
        additional_input_args = additional_input_args if additional_input_args is not None else []
        self._validate_input_args([num_cont_col, cat_sizes, *additional_input_args])

        # Call of parent's constructor
        Module.__init__(self)

        # Settings of general protected attributes
        self._alpha = alpha
        self._beta = beta
        self._criterion = criterion
        self._evaluations: Dict[str, Evaluation] = {}
        self._input_size = num_cont_col if num_cont_col is not None else 0
        self._path_to_model = path_to_model
        self._optimizer = None
        self._output_size = output_size
        self._verbose = verbose

        # Settings of protected attributes related to entity embedding
        self._cat_idx = cat_idx if cat_idx is not None else []
        self._cont_idx = list(range(num_cont_col))
        self._embedding_block = None

        # Initialization of a protected method
        self._update_weights = None

        # # We set the embedding layers
        # if len(cat_idx) != 0 and cat_sizes is not None:
        #
        #     # We check embedding sizes (if nothing provided -> emb_sizes = cat_sizes - 1)
        #     if cat_emb_sizes is None:
        #         cat_emb_sizes = [s - 1 for s in cat_sizes]
        #         if 0 in cat_emb_sizes:
        #             raise ValueError('One categorical variable as a single modality')
        #
        #     # We create the embedding layers
        #     self._embedding_block = EntityEmbeddingBlock(cat_sizes, cat_emb_sizes, cat_idx)
        #
        #     # We sum the length of all embeddings
        #     self._input_size += self._embedding_block.output_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def _init_evaluations_dictionary(self) -> None:
        """
        Initialize evaluations dictionary.
        """
        tasks = self._criterion.tasks

        for i in [MaskType.TRAIN, MaskType.VALID]:
            self._evaluations[i] = Evaluation(
                losses=dict(**{self._criterion.name: []}, **{task.criterion.name: [] for task in tasks}),
                scores={task.metric.name: [] for task in tasks}
            )

    @staticmethod
    def _create_validation_loader(
            dataset: ProstateCancerDataset,
            valid_batch_size: Optional[int]
    ) -> DataLoader:
        """
        Creates the objects needed for validation during the training process.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloader.
        valid_batch_size : Optional[int]
            Size of the batches in the valid loader (None = one single batch).

        Returns
        -------
        validation_loader : DataLoader
            Validation loader.
        """
        # We create the valid dataloader (if valid size != 0)
        valid_size, valid_data, early_stopper = len(dataset.valid_mask), None, None

        if valid_size != 0:

            # We check if a valid batch size was provided
            valid_batch_size = min(valid_size, valid_batch_size) if valid_batch_size is not None else valid_size

            # We create the valid loader
            valid_data = DataLoader(
                dataset,
                batch_size=valid_batch_size,
                sampler=SubsetRandomSampler(dataset.valid_mask)
            )

        return valid_data

    def _disable_running_stats(self) -> None:
        """
        Disables batch norm momentum when executing SAM optimization step.
        """
        self.apply(self.disable_module_running_stats)

    def _enable_running_stats(self) -> None:
        """
        Restores batch norm momentum when executing SAM optimization step
        """
        self.apply(self.enable_module_running_stats)

    def _sam_weight_update(
            self,
            x: List[Union[DGLGraph, Tensor]],
            y: Tensor,
            pos_idx: Optional[List[int]] = None
    ) -> Tuple[Tensor, float]:
        """
        Executes a weights update using Sharpness-Aware Minimization (SAM) optimizer.

        Note from https://github.com/davda54/sam :
            The running statistics are computed in both forward passes, but they should be computed only for the
            first one. A possible solution is to set BN momentum to zero to bypass the running statistics during the
            second pass.

        Parameters
        ----------
        x : Union[DGLGraph, Tensor]
            A list of arguments taken for the forward pass (DGLGraph and (N', D) tensor with batch inputs).
        y : Tensor
            (N',) ground truth associated to a batch.
        pos_idx : Optional[List[int]]
            Dictionary that maps the original dataset's idx to their current position in the mask used for the forward
            pass (used only with GNNs)

        Returns
        -------
        (pred, loss) : Tuple[Tensor, float]
            Tuple of a (N',) tensor with predictions and training loss.
        """
        # We compute the predictions
        pred = self(*x)
        pred = pred if pos_idx is None else pred[pos_idx]

        # First forward-backward pass
        loss = self.loss(pred, y)
        loss.backward()
        self._optimizer.first_step()

        # Second forward-backward pass
        self._disable_running_stats()
        second_pred = self(*x)
        second_pred = second_pred if pos_idx is None else second_pred[pos_idx]
        self.loss(second_pred, y).backward()
        self._optimizer.second_step()

        # We enable running stats again
        self._enable_running_stats()

        return pred, loss.item()

    def _basic_weight_update(
            self,
            x: List[Union[DGLGraph, Tensor]],
            y: Tensor,
            pos_idx: Optional[List[int]] = None
    ) -> Tuple[Tensor, float]:
        """
        Executes a weights update without using Sharpness-Aware Minimization (SAM).

        Parameters
        ----------
        x : List[Union[DGLGraph, Tensor]]
            A list of arguments taken for the forward pass (DGLGraph and (N', D) tensor with batch inputs).
        y : Tensor
            (N',) ground truth associated to a batch.
        pos_idx : Optional[List[int]]
            Dictionary that maps the original dataset's idx to their current position in the mask used for the forward
            pass (used only with GNNs).

        Returns
        -------
        (pred, loss) : Tuple[Tensor, float]
            Tuple of a (N',) tensor with predictions and training loss.
        """
        # We compute the predictions
        pred = self(*x)
        pred = pred if pos_idx is None else pred[pos_idx]

        # We execute a single forward-backward pass
        loss = self.loss(pred, y)
        loss.backward()
        self._optimizer.step()

        return pred, loss.item()

    def _generate_progress_func(
            self,
            max_epochs: int
    ) -> Callable:
        """
        Builds a function that updates the training progress in the terminal.

        Parameters
        ----------
        max_epochs : int
            Maximum number of training epochs.

        Returns
        -------
        progress_function : Callable
            update_progress
        """
        if self._verbose:
            def update_progress(epoch: int, mean_epoch_loss: float):
                if (epoch + 1) % 5 == 0 or (epoch + 1) == max_epochs:
                    print(f"Epoch {epoch + 1} - Loss : {round(mean_epoch_loss, 4)}")
        else:
            def update_progress(*args):
                pass

        return update_progress

    def _update_evaluations_progress(
            self,
            losses: Dict[str, float],
            scores: Dict[str, float],
            mask_type: str
    ) -> Dict[str, float]:
        """
        Adds epoch score and loss to the evaluations history.

        Parameters
        ----------
        losses: Dict[str, float]
            Epoch losses.
        scores: Dict[str, float]
            Epoch scores.
        mask_type : str
            "train" of "valid".

        Returns
        -------
        losses or scores : Dict[str, float]
            Mean epoch losses or mean epoch scores.
        """
        # We update the evaluations history
        for name, loss in losses.items():
            self._evaluations[mask_type].losses[name].append(loss)

        for name, score in scores.items():
            self._evaluations[mask_type].scores[name].append(score)

        # We return a value according to the mask type
        if mask_type == MaskType.VALID:
            return scores

        return losses

    def fit(
            self,
            dataset: ProstateCancerDataset,
            early_stopper: EarlyStopper,
            lr: float,
            rho: float = 0,
            batch_size: Optional[int] = 55,
            valid_batch_size: Optional[int] = None,
            max_epochs: int = 200
    ) -> None:
        """
        Fits the model to the training data.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            Prostate cancer dataset used to feed the dataloaders.
        early_stopper : EarlyStopper
            Early stopper.
        lr : float
            Learning rate
        rho : float
            If rho >= 0, will be used as neighborhood size in Sharpness-Aware Minimization optimizer, otherwise, Adam
            optimizer will be used.
        batch_size : Optional[int]
            Size of the batches in the training loader.
        valid_batch_size : Optional[int]
            Size of the batches in the valid loader (None = one single batch).
        max_epochs : int
            Maximum number of epochs for training.
        """
        # We assume that the tasks in the dataset are the tasks on which we need to calculate the loss.
        self._criterion.tasks = dataset.tasks

        # We setup the early stopper depending on its type.
        if early_stopper.early_stopper_type == EarlyStopperType.METRIC:
            early_stopper.tasks = dataset.tasks
        elif early_stopper.early_stopper_type == EarlyStopperType.LOSS:
            early_stopper.criterion = self._criterion

        # We create an empty evaluations dictionary that logs losses and metrics values.
        self._init_evaluations_dictionary()

        # We create the training objects
        train_data = self._create_train_dataloader(dataset, batch_size)

        # We create the objects needed for validation (data loader, early stopper)
        valid_data = self._create_validation_loader(dataset, valid_batch_size)

        # We init the update function
        update_progress = self._generate_progress_func(max_epochs)

        # We set the optimizer
        if rho > 0:
            self._update_weights = self._sam_weight_update
            self._optimizer = SAM(self.parameters(), Adam, rho=rho, lr=lr)
        else:
            self._update_weights = self._basic_weight_update
            self._optimizer = Adam(self.parameters(), lr=lr)

        # # We add the dataset to train_data and valid_data if it is a GNN dataset
        # if isinstance(dataset, PetaleKGNNDataset):
        #     train_data = (train_data, dataset)
        #     valid_data = (valid_data, dataset)

        # We execute the epochs
        for epoch in range(max_epochs):

            # We calculate training loss
            train_loss = self._execute_train_step(train_data)
            update_progress(epoch, train_loss)

            # We calculate valid score and apply early stopping if needed
            if self._execute_valid_step(valid_data, early_stopper):
                early_stopper.print_early_stopping_message(epoch)
                break

        if early_stopper is not None:

            # We extract best params and remove checkpoint file
            self.load_state_dict(early_stopper.get_best_params())
            early_stopper.remove_checkpoint()

    def loss(
            self,
            pred: Tensor,
            y: Tensor
    ) -> Tensor:
        """
        Calls the criterion and add the elastic penalty.

        Parameters
        ----------
        pred : Tensor
            (N, C) tensor if classification with C classes, (N,) tensor for regression.
        y : Tensor
            (N,) tensor with targets.
        Returns
        -------
        loss : Tensor
            Tensor with loss value.
        """
        # Computations of penalties
        l1_penalty, l2_penalty = tensor(0.), tensor(0.)
        for _, w in self.named_parameters():
            l1_penalty = l1_penalty + w.abs().sum()
            l2_penalty = l2_penalty + w.pow(2).sum()

        # Computation of loss reduction + elastic penalty
        return self._criterion(pred, y.float()) + self._alpha * l1_penalty + self._beta * l2_penalty

    @abstractmethod
    def predict(
            self,
            x: DataModel.x
    ) -> DataModel.y:
        """
        Returns predictions for all samples in a particular batch. For classification tasks, it returns the probability
        of belonging to class 1. For regression tasks, it returns the predicted real-valued target. For segmentation
        tasks, it returns the predicted segmentation map.

        Parameters
        ----------
        x : DataElement.x
            Batch data items.

        Returns
        -------
        predictions : DataModel.y
            Predictions.
        """
        raise NotImplementedError

    def plot_evaluations(
            self,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plots the training and valid curves saved.

        Parameters
        ----------
        save_path : Optional[str]
            Path were the figures will be saved.
        """
        train_history, valid_history, progression_type = [], [], []
        for train, valid in zip(self._evaluations[MaskType.TRAIN], self._evaluations[MaskType.VALID]):
            for name, train_loss in train.losses.items():
                train_history.append(train_loss)
                valid_history.append(valid.losses[name])
                progression_type.append(name)
            for name, train_score in train.scores.items():
                train_history.append(train_score)
                valid_history.append(valid.scores[name])
                progression_type.append(name)

        # Figure construction
        visualize_epoch_progression(
            train_history=train_history,
            valid_history=valid_history,
            progression_type=progression_type,
            path=save_path
        )

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
            drop_last=(train_size % batch_size) == 1
        )

        return train_data

    @staticmethod
    def _validate_input_args(
            input_args: List[Any]
    ) -> None:
        """
        Checks if all arguments related to inputs are None. If not, the inputs are valid.

        Parameters
        ----------
        input_args : List[Any]
            A list of arguments related to inputs.
        """
        valid = False
        for arg in input_args:
            if arg is not None:
                valid = True
                break

        if not valid:
            raise ValueError("There must be continuous columns or categorical columns")

    @abstractmethod
    def _execute_train_step(
            self,
            train_data: Union[DataLoader, Tuple[DataLoader, ProstateCancerDataset]]
    ) -> float:
        """
        Executes one training epoch.

        Parameters
        ----------
        train_data : Union[DataLoader, Tuple[DataLoader, ProstateCancerDataset]]
            Training dataloader or tuple (train loader, dataset).

        Returns
        -------
        loss : float
            Mean epoch loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _execute_valid_step(
            self,
            valid_data: Optional[Union[DataLoader, Tuple[DataLoader, ProstateCancerDataset]]],
            early_stopper: EarlyStopper
    ) -> bool:
        """
        Executes an inference step on the validation data.

        Parameters
        ----------
        valid_data : Optional[Union[DataLoader, Tuple[DataLoader, ProstateCancerDataset]]]
            Valid dataloader or tuple (valid loader, dataset).
        early_stopper : EarlyStopper
            Early stopper.

        Returns
        -------
        stop : bool
            True if we need to early stop.
        """
        raise NotImplementedError

    @staticmethod
    def disable_module_running_stats(
            module: Module
    ) -> None:
        """
        Sets momentum to 0 for all BatchNorm layer in the module after saving it in a cache.

        Parameters
        ----------
        module: torch module
        """
        if isinstance(module, BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0

    @staticmethod
    def enable_module_running_stats(
            module: Module
    ) -> None:
        """
        Restores momentum for all BatchNorm layer in the module using the value in the cache.

        Parameters
        ----------
        module : Module
            Torch module
        """
        if isinstance(module, BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
