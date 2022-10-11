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
import os
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

# from dgl import DGLGraph
from monai.data import DataLoader
import numpy as np
from torch import FloatTensor, no_grad, stack, tensor, Tensor
from torch.nn import BatchNorm1d, Module
from torch.optim import Adam
from torch.utils.data import SubsetRandomSampler

from src.data.datasets.prostate_cancer_dataset import DataModel, ProstateCancerDataset
from src.data.processing.tools import MaskType
# from src.data.processing.gnn_datasets import PetaleKGNNDataset
from src.models.base.blocks.embeddings import EntityEmbeddingBlock
from src.training.early_stopper import EarlyStopper, EarlyStopperType, MetricEarlyStopper, MultiTaskLossEarlyStopper
from src.training.optimizer import SAM
from src.utils.multi_task_losses import MultiTaskLoss
from src.utils.reductions import MetricReduction
from src.utils.score_metrics import Direction
from src.utils.tasks import Task, TaskType
from src.visualization.tools import visualize_epoch_progression


class Output(NamedTuple):
    predictions: List = []
    targets: List = []


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
            calculate_epoch_score: bool = True,
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
        calculate_epoch_score : bool
            Whether we want to calculate the score at each training epoch.
        verbose : bool
            True if we want to print the training progress.
        """
        # Call of parent's constructor
        Module.__init__(self)

        # Settings of general protected attributes
        self._alpha = alpha
        self._beta = beta
        self._calculate_epoch_score = calculate_epoch_score
        self._criterion = criterion
        self._dataset: Optional[ProstateCancerDataset] = None
        self._evaluations: Dict[str, Evaluation] = {}
        self._path_to_model = path_to_model
        self._optimizer = None
        self._output_size = output_size
        self._tasks = None
        self._verbose = verbose

        # Create model path
        os.makedirs(path_to_model, exist_ok=True)

        # Initialization of a protected method
        self._update_weights = None

    @property
    def embedding_block(self) -> EntityEmbeddingBlock:
        embedding_block = None

        # We set the embedding layers
        if len(self._dataset.table_dataset.cat_idx) != 0 and self._dataset.table_dataset.cat_sizes is not None:

            # We check embedding sizes (if nothing provided -> emb_sizes = cat_sizes - 1)
            cat_emb_sizes = [s - 1 for s in self._dataset.table_dataset.cat_sizes]
            if 0 in cat_emb_sizes:
                raise ValueError('One categorical variable as a single modality')

            embedding_block = EntityEmbeddingBlock(
                cat_sizes=self._dataset.table_dataset.cat_sizes,
                cat_emb_sizes=cat_emb_sizes,
                cat_idx=self._dataset.table_dataset.cat_idx
            )

        return embedding_block

    @property
    def table_input_size(self) -> int:
        if self.embedding_block:
            return len(self._dataset.table_dataset.cont_cols) + self.embedding_block.output_size
        else:
            return len(self._dataset.table_dataset.cont_cols)

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def tasks(self) -> Optional[List[Task]]:
        return self._tasks

    def _init_evaluations_dictionary(self) -> None:
        """
        Initialize evaluations dictionary.
        """
        for i in [MaskType.TRAIN, MaskType.VALID]:
            self._evaluations[i] = Evaluation(
                losses=dict(**{self._criterion.name: []}, **{task.name: [] for task in self._tasks}),
                scores={task.name: [] for task in self._tasks}
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
        valid_size, valid_data = len(dataset.valid_mask), None

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
            x: DataModel.x,
            y: DataModel.y
    ) -> Tuple[Dict[str, Tensor], float]:
        """
        Executes a weights update using Sharpness-Aware Minimization (SAM) optimizer.

        Note from https://github.com/davda54/sam :
            The running statistics are computed in both forward passes, but they should be computed only for the
            first one. A possible solution is to set BN momentum to zero to bypass the running statistics during the
            second pass.

        Parameters
        ----------
        x : DataModel.x
            Batch data items.
        y : DataModel.y
            Batch data items.

        Returns
        -------
        (pred, loss) : Tuple[DataModel.y, float]
            Tuple of a dictionary of tensors with predictions and training loss.
        """
        # We compute the predictions
        pred = self(x)

        # First forward-backward pass
        loss = self.loss(pred, y)
        loss.backward()
        self._optimizer.first_step()

        # Second forward-backward pass
        self._disable_running_stats()
        second_pred = self(x)
        self.loss(second_pred, y).backward()
        self._optimizer.second_step()

        # We enable running stats again
        self._enable_running_stats()

        return pred, loss.item()

    def _basic_weight_update(
            self,
            x: DataModel.x,
            y: DataModel.y
    ) -> Tuple[Dict[str, Tensor], float]:
        """
        Executes a weights update without using Sharpness-Aware Minimization (SAM).

        Parameters
        ----------
        x : DataModel.x
            Batch data items.
        y : DataModel.y
            Batch data items.

        Returns
        -------
        (pred, loss) : Tuple[DataModel.y, float]
            Tuple of a dictionary of tensors with predictions and training loss.
        """
        # We compute the predictions
        pred = self(x)

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
    ):
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
        """
        # We update the evaluations history
        for name, loss in losses.items():
            self._evaluations[mask_type].losses[name].append(loss)

        if self._calculate_epoch_score:
            for name, score in scores.items():
                self._evaluations[mask_type].scores[name].append(score)

    def fit(
            self,
            dataset: ProstateCancerDataset,
            early_stopper_type: EarlyStopperType,
            lr: float,
            patience: int = 10,
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
        early_stopper_type : EarlyStopperType
            Early stopper type.
        lr : float
            Learning rate
        rho : float
            If rho >= 0, will be used as neighborhood size in Sharpness-Aware Minimization optimizer, otherwise, Adam
            optimizer will be used.
        patience : float
            Patience.
        batch_size : Optional[int]
            Size of the batches in the training loader.
        valid_batch_size : Optional[int]
            Size of the batches in the valid loader (None = one single batch).
        max_epochs : int
            Maximum number of epochs for training.
        """
        # We set the dataset
        self._dataset = dataset

        # We assume that the tasks in the dataset are the tasks on which we need to calculate the loss.
        self._tasks, self._criterion.tasks = dataset.tasks, dataset.tasks

        # We execute the callbacks that need to be ran at the beginning of training.
        self.on_fit_begin()

        # We setup the early stopper depending on its type.
        early_stopper = None
        if early_stopper_type == EarlyStopperType.METRIC:
            early_stopper = MetricEarlyStopper(path_to_model=self._path_to_model, patience=patience)
            early_stopper.tasks = dataset.tasks
        elif early_stopper_type == EarlyStopperType.MULTITASK_LOSS:
            early_stopper = MultiTaskLossEarlyStopper(path_to_model=self._path_to_model, patience=patience)
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
            pred: DataModel.y,
            y: DataModel.y,
    ) -> Tensor:
        """
        Calls the criterion and add the elastic penalty.

        Parameters
        ----------
        pred : DataModel.y
            Predictions.
        y: DataModel.y
            Targets.

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
        return self._criterion(pred, y) + self._alpha * l1_penalty + self._beta * l2_penalty

    @abstractmethod
    def on_fit_begin(self):
        """
        Called when the training (fit) phase starts.
        """
        raise NotImplementedError

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

    def predict_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int],
    ) -> DataModel.y:
        """
        Returns predictions for all samples in a particular subset of the dataset, determined using a mask parameter.
        For classification tasks, it returns the probability of belonging to class 1. For regression tasks, it returns
        the predicted real-valued target.

        NOTE : It doesn't return segmentation map as it will bust the computer's RAM.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the predictions.

        Returns
        -------
        predictions : DataModel.y
            Predictions (except segmentation map).
        """
        subset = dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        predictions = {}
        with no_grad():
            for idx, (x, _) in enumerate(data_loader):
                pred = self.predict(x)

                for task in dataset.tasks:
                    if task.task_type != TaskType.SEGMENTATION:
                        if idx == 0:
                            predictions[task.name] = pred[task.name]
                        else:
                            predictions[task.name] = stack([predictions[task.name], pred[task.name]], dim=0)

        return predictions

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
        train = self._evaluations[MaskType.TRAIN]
        valid = self._evaluations[MaskType.VALID]

        for name, train_loss in train.losses.items():
            train_history.append(train_loss)
            valid_history.append(valid.losses[name])
            progression_type.append(name)

        if self._calculate_epoch_score:
            for name, train_score in train.scores.items():
                train_history.append(train_score)
                valid_history.append(valid.scores[name])
                progression_type.append(name)

        # Figure construction
        visualize_epoch_progression(
            train_history=train_history,
            valid_history=valid_history,
            progression_type=progression_type,
            path=save_path if save_path else self._path_to_model
        )

    def scores(
            self,
            predictions: DataModel.y,
            targets: DataModel.y,
            include_evaluation_metrics: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the scores for all samples in a particular batch.

        Parameters
        ----------
        predictions : DataModel.y
            Batch data items.
        targets : DataElement.y
            Batch data items.
        include_evaluation_metrics: bool
            Whether to calculate the scores with the evaluation metrics or not.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        with no_grad():
            scores = {}
            for task in self._tasks:
                scores[task.name] = {}
                metrics = [task.optimization_metric]

                if include_evaluation_metrics and task.evaluation_metrics:
                    metrics = metrics + task.evaluation_metrics

                for metric in metrics:
                    scores[task.name][metric.name] = metric(
                        np.array(predictions[task.name]), np.array(targets[task.name])
                    )

        return scores

    def scores_dataset(
            self,
            dataset: ProstateCancerDataset,
            mask: List[int],
            include_evaluation_metrics: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Returns the score of all samples in a particular subset of the dataset, determined using a mask parameter.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        mask : List[int]
            A list of dataset idx for which we want to obtain the mean score.
        include_evaluation_metrics: bool
            Whether to calculate the scores with the evaluation metrics or not.

        Returns
        -------
        scores : Dict[str, Dict[str, float]]
            Score for each tasks and each metrics.
        """
        subset = dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        scores = {task.name: {} for task in self.tasks}
        segmentation_scores_dict = {}
        for task in self.tasks:
            if task.task_type == TaskType.SEGMENTATION:
                segmentation_scores_dict[task.name] = {}

                metrics = [task.optimization_metric]
                if include_evaluation_metrics and task.evaluation_metrics:
                    metrics = metrics + task.evaluation_metrics

                for metric in metrics:
                    segmentation_scores_dict[task.name][metric.name] = []

        non_segmentation_outputs_dict = {
            task.name: Output() for task in self.tasks if task.task_type != TaskType.SEGMENTATION
        }

        # Set model for evaluation
        self.eval()

        with no_grad():
            for x, targets in data_loader:
                predictions = self.predict(x)

                for task in self.tasks:
                    pred, target = predictions[task.name].item(), targets[task.name].item()

                    if task.task_type == TaskType.SEGMENTATION:
                        metrics = [task.optimization_metric]
                        if include_evaluation_metrics and task.evaluation_metrics:
                            metrics = metrics + task.evaluation_metrics

                        for metric in metrics:
                            segmentation_scores_dict[task.name][metric.name].append(
                                metric(np.array(pred), np.array(target), MetricReduction.NONE)
                            )
                    else:
                        non_segmentation_outputs_dict[task.name].predictions.append(pred)
                        non_segmentation_outputs_dict[task.name].targets.append(target)

            for task in self.tasks:
                metrics = [task.optimization_metric]
                if include_evaluation_metrics and task.evaluation_metrics:
                    metrics = metrics + task.evaluation_metrics

                if task.task_type == TaskType.SEGMENTATION:
                    for metric in metrics:
                        scores[task.name][metric.name] = metric.perform_reduction(
                            FloatTensor(segmentation_scores_dict[task.name][metric.name])
                        )
                else:
                    output = non_segmentation_outputs_dict[task.name]
                    for metric in metrics:
                        scores[task.name][metric.name] = metric(np.array(output.predictions), np.array(output.targets))

        return scores

    def fix_thresholds_to_optimal_values(
            self,
            dataset: ProstateCancerDataset,
            include_evaluation_metrics: bool = False
    ) -> None:
        """
        Fix all classification thresholds to their optimal values according to a given metric.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.
        include_evaluation_metrics: bool
            Whether to fix the thresholds of evaluation metrics or not.
        """
        subset = dataset[dataset.train_mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False)

        thresholds = np.linspace(start=0.01, stop=0.95, num=95)

        classification_tasks = [task for task in self.tasks if task.task_type == TaskType.CLASSIFICATION]
        outputs_dict = {task.name: Output() for task in classification_tasks}

        # Set model for evaluation
        self.eval()

        with no_grad():
            for x, targets in data_loader:
                predictions = self.predict(x)

                for task in classification_tasks:
                    pred, target = predictions[task.name].item(), targets[task.name].item()

                    outputs_dict[task.name].predictions.append(pred)
                    outputs_dict[task.name].targets.append(target)

            for task in classification_tasks:
                output = outputs_dict[task.name]
                metrics = [task.optimization_metric]

                if include_evaluation_metrics and task.evaluation_metrics:
                    metrics = metrics + task.evaluation_metrics

                for metric in metrics:
                    scores = np.array(
                        [metric(np.array(output.predictions), np.array(output.targets), t) for t in thresholds]
                    )

                    # We set the threshold to the optimal threshold
                    if metric.direction == Direction.MINIMIZE.value:
                        metric.threshold = thresholds[np.argmin(scores)]
                    else:
                        metric.threshold = thresholds[np.argmax(scores)]

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
