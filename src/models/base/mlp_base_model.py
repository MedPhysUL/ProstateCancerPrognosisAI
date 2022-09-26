"""
    @file:              mlp_base_model.py
    @Author:            Maxence Larose, Nicolas Raymond, Mehdi Mitiche

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define an MLP model with entity embeddings. These models are not shaped
                        to inherit from the BaseModel class. However, two wrapper classes for torch models are provided
                        to enable the use of these mlp models with hyperparameter tuning functions.
"""

from typing import List, Optional

import numpy as np
from monai.data import DataLoader
from torch import cat, no_grad, sigmoid
from torch.nn import Identity, Linear

from src.data.datasets.prostate_cancer_dataset import DataModel
from src.data.processing.tools import MaskType
from src.models.base.custom_torch_base import TorchCustomModel
from src.models.base.blocks.encoders import MLPEncodingBlock
from src.training.early_stopper import EarlyStopper, EarlyStopperType
from src.utils.multi_task_losses import MultiTaskLoss
from src.utils.tasks import TaskType


class MLPBaseModel(TorchCustomModel):
    """
    Multilayer perceptron model with entity embedding for categorical variables.
    """

    def __init__(
            self,
            output_size: int,
            layers: List[int],
            activation: str,
            criterion: MultiTaskLoss,
            path_to_model: str,
            dropout: float = 0,
            alpha: float = 0,
            beta: float = 0,
            calculate_epoch_score: bool = True,
            verbose: bool = False
    ):
        """
        Builds the layers of the model and sets other protected attributes.

        Parameters
        ----------
        output_size : int
            Number of nodes in the last layer of the neural network
        layers : List[int]
            List with number of units in each hidden layer
        criterion : MultiTaskLoss
            Loss function of our model
        path_to_model : str
            Path to save model.
        activation : str
            Activation function
        dropout : float
            Probability of dropout
        alpha : float
            L1 penalty coefficient
        beta : float
            L2 penalty coefficient
        calculate_epoch_score : bool
            Whether we want to calculate the score at each training epoch.
        verbose : bool
            True if we want trace of the training progress
        """

        # We call parent's constructor
        super().__init__(
            criterion=criterion,
            output_size=output_size,
            path_to_model=path_to_model,
            alpha=alpha,
            beta=beta,
            calculate_epoch_score=calculate_epoch_score,
            verbose=verbose
        )

        if len(layers) > 0:
            self._main_encoding_block = MLPEncodingBlock(
                input_size=self._input_size,
                output_size=layers[-1],
                layers=layers[:-1],
                activation=activation,
                dropout=dropout
            )
        else:
            self._main_encoding_block = Identity()
            layers.append(self._input_size)

        # We add a linear layer to complete the layers
        self._linear_layer = Linear(layers[-1], output_size)

    def _execute_train_step(
            self,
            train_data: DataLoader
    ) -> float:
        """
        Executes one training epoch.

        Parameters
        ----------
        train_data : DataLoader
            Training dataloader

        Returns
        -------
        Mean epoch loss
        """
        # We set the model for training
        self.train()
        epoch_losses = dict(**{self._criterion.name: []}, **{task.criterion.name: [] for task in self._tasks})

        # We execute one training step
        for x, y in train_data:
            # We clear the gradients
            self._optimizer.zero_grad()

            # We perform the weight update
            pred, loss = self._update_weights([x], y)

            # We update the losses history
            epoch_losses[self._criterion.name].append(loss.item())
            for name, single_task_loss in self._criterion.single_task_losses:
                epoch_losses[name].append(single_task_loss)

        epoch_losses = {name: np.mean(loss) for name, loss in epoch_losses.items()}

        if self._calculate_epoch_score:
            # We get the scores values
            self.fix_thresholds_to_optimal_values(dataset=self._dataset)
            scores = self.scores_dataset(dataset=self._dataset, mask=self._dataset.train_mask)
            epoch_scores = {
                f"{task.name}_{task.optimization_metric.name}": scores[task.name][task.optimization_metric.name]
                for task in self._dataset.tasks
            }
        else:
            epoch_scores = {}

        # We update evaluations history
        self._update_evaluations_progress(
            losses=epoch_losses,
            scores=epoch_scores,
            mask_type=MaskType.TRAIN
        )

        return epoch_losses[self._criterion.name]

    def _execute_valid_step(
            self,
            valid_loader: Optional[DataLoader],
            early_stopper: EarlyStopper
    ) -> bool:
        """
        Executes an inference step on the validation data.

        Parameters
        ----------
        valid_loader : Optional[DataLoader]
            Validation dataloader

        Returns
        -------
        True if we need to early stop
        """
        if valid_loader is None:
            return False

        # Set model for evaluation
        self.eval()
        epoch_losses = dict(**{self._criterion.name: []}, **{task.criterion.name: [] for task in self._tasks})

        # We execute one inference step on validation set
        with no_grad():

            for x, y in valid_loader:
                # We perform the forward pass
                output = self(x)

                # We calculate the loss and the score
                loss = self.loss(output, y)

                # We update the losses history
                epoch_losses[self._criterion.name].append(loss.item())
                for name, single_task_loss in self._criterion.single_task_losses:
                    epoch_losses[name].append(single_task_loss)

        epoch_losses = {name: np.mean(loss) for name, loss in epoch_losses.items()}

        if self._calculate_epoch_score:
            scores = self.scores_dataset(dataset=self._dataset, mask=self._dataset.valid_mask)
            epoch_scores = {
                f"{task.name}_{task.optimization_metric.name}": scores[task.name][task.optimization_metric.name]
                for task in self._dataset.tasks
            }
        else:
            epoch_scores = {}

        # We update evaluations history
        self._update_evaluations_progress(
            losses=epoch_losses,
            scores=epoch_scores,
            mask_type=MaskType.VALID
        )

        # We check early stopping status based on its type.
        if early_stopper.early_stopper_type == EarlyStopperType.MULTITASK_LOSS:
            early_stopper(epoch_losses[self._criterion.name], self)
        elif early_stopper.early_stopper_type == EarlyStopperType.METRIC:
            early_stopper(list(epoch_scores.values()), self)

        if early_stopper.early_stop:
            return True

        return False

    def forward(
            self,
            x: DataModel.x
    ) -> DataModel.y:
        """
        Executes the forward pass.

        Parameters
        ----------
        x : DataElement.x
            Batch data items.

        Returns
        -------
        predictions : DataModel.y
            Predictions.
        """
        # We retrieve the table data only and transform the input dictionary to a tensor
        x_table = cat(list(x.table.values()), 1)

        # We initialize a list of tensors to concatenate
        new_x_table = []

        # We extract continuous data
        if len(self._dataset.table_dataset.cont_idx) != 0:
            new_x_table.append(x_table[:, self._dataset.table_dataset.cont_idx])

        # We perform entity embeddings on categorical features
        if len(self._dataset.table_dataset.cat_idx) != 0:
            new_x_table.append(self._embedding_block(x_table))

        # We concatenate all inputs
        x_table = cat(new_x_table, 1)

        # We compute the output
        y_table = self._linear_layer(self._main_encoding_block(x_table)).squeeze()

        y = {task.name: y_table[i] for i, task in enumerate(self.tasks)}

        return y

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
        # Set model for evaluation
        self.eval()

        predictions = {}
        with no_grad():
            features = cat(list(x.table.values()), dim=1)
            outputs = self(features)

            for task in self.tasks:
                if task.task_type == TaskType.CLASSIFICATION:
                    predictions[task.name] = sigmoid(outputs[task.name])
                elif task.task_type == TaskType.REGRESSION:
                    predictions[task.name] = outputs[task.name]

        return predictions
