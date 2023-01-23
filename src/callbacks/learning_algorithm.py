"""
    @file:              learning_algorithm.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 12/2022

    @Description:       This file is used to define a 'LearningAlgorithm' which dictates how to update the model's
                        parameters during the training process.
"""

from typing import Dict, Iterable, Optional, Union

from torch import Tensor
from torch.optim import Optimizer

from src.callbacks.callback import Callback, Priority
from src.callbacks.model_checkpoint import ModelCheckpoint
from src.callbacks.utils.regularization import Regularization, RegularizationList
from src.utils.multi_task_losses import MultiTaskLoss


class LearningAlgorithm(Callback):

    def __init__(
            self,
            criterion: MultiTaskLoss,
            optimizer: Optimizer,
            regularization: Optional[Union[Regularization, RegularizationList, Iterable[Regularization]]] = None,
            **kwargs
    ):
        """
        Constructor for 'LearningAlgorithm' class.

        Parameters
        ----------
        criterion : MultiTaskLoss
            Multi-task loss.
        optimizer : Optimizer
            A pytorch Optimizer.
        regularization : Optional[Union[Regularization, RegularizationList, Iterable[Regularization]]]
            Regularization.
        """
        super().__init__(**kwargs)

        self.criterion = criterion
        self.optimizer = optimizer
        self.regularization = regularization

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.MEDIUM_PRIORITY.value

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'CallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
        """
        return True

    def load_checkpoint_state(self, trainer, checkpoint: dict, **kwargs):
        """
        Loads the state of the callback from a dictionary.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        checkpoint : dict
            The dictionary containing all the states of the trainer.
        """
        if self.save_state:
            state = checkpoint.get(self.name, {})
            optimizer_state = state.get(ModelCheckpoint.CHECKPOINT_OPTIMIZER_STATE_KEY, None)
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

    def get_checkpoint_state(self, trainer, **kwargs) -> Optional[dict]:
        """
        Get the state of the callback.

        Parameters
        ----------
        trainer : Trainer
            The trainer.

        Returns
        -------
        state : Optional[dict]
            The state of the callback.
        """
        if self.save_state:
            if self.optimizer is not None:
                return {
                    ModelCheckpoint.CHECKPOINT_OPTIMIZER_STATE_KEY: self.optimizer.state_dict()
                }

        return None

    def _update_batch_state(
            self,
            single_task_losses: Dict[str, Tensor],
            multi_task_loss_without_regularization: Tensor,
            multi_task_loss_with_regularization: Optional[Tensor],
            trainer
    ):
        single_task_losses_dict = {
            task.name: {
                task.criterion.name: single_task_losses[task.name].detach().item()
            } for task in self.criterion.tasks
        }

        multi_task_losses_without_regularization_dict = {
            self.name: {self.criterion.name: multi_task_loss_without_regularization.detach().item()}
        }

        if multi_task_loss_with_regularization:
            multi_task_losses_with_regularization_dict = {
                self.name: {self.criterion.name: multi_task_loss_with_regularization.detach().item()}
            }
        else:
            multi_task_losses_with_regularization_dict = {}

        trainer.batch_state.single_task_losses = {
            **trainer.batch_state.single_task_losses,
            **single_task_losses_dict
        }

        trainer.batch_state.multi_task_losses_with_regularization = {
            **trainer.batch_state.multi_task_losses_with_regularization,
            **multi_task_losses_with_regularization_dict
        }

        trainer.batch_state.multi_task_losses_without_regularization = {
            **trainer.batch_state.multi_task_losses_without_regularization,
            **multi_task_losses_without_regularization_dict
        }

    def _compute_loss(self, pred_batch: Dict[str, Tensor], y_batch: Dict[str, Tensor], trainer) -> Tensor:
        """
        Calls the criterion and add the elastic penalty.

        Parameters
        ----------
        pred_batch : Dict[str, Tensor]
            Predictions.
        y_batch : Dict[str, Tensor]
            Targets.
        trainer : Trainer
            The trainer.

        Returns
        -------
        batch_loss : Tensor
            Tensor with loss value.
        """
        losses = {task.name: task.criterion(pred_batch[task.name], y_batch[task.name]) for task in self.criterion.tasks}

        loss_without_regularization = self.criterion(losses)

        loss_with_regularization = None
        if self.regularization:
            loss_with_regularization = loss_without_regularization + self.regularization()

        self._update_batch_state(losses, loss_without_regularization, loss_with_regularization, trainer)

        return loss_with_regularization if loss_with_regularization else loss_without_regularization

    def on_fit_start(self, trainer, **kwargs):
        """
        Sets criterion tasks.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keyword arguments.
        """
        tasks = trainer.training_state.tasks

        if self.criterion.tasks is None:
            self.criterion.tasks = tasks

    def _optimizer_step(self, pred_batch: Dict[str, Tensor], y_batch: Dict[str, Tensor], trainer):
        """
        Perform an optimizer step, i.e calculate loss and perform backward pass.

        Parameters
        ----------
        pred_batch : Dict[str, Tensor]
            Predictions.
        y_batch : Dict[str, Tensor]
            Targets.
        trainer : Trainer
            The trainer.

        Returns
        -------
        batch_loss : Tensor
            Tensor with loss value.
        """
        self.optimizer.zero_grad()
        batch_loss = self._compute_loss(pred_batch, y_batch, trainer)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss.detach_()

    def on_optimization_start(self, trainer, **kwargs):
        """
        Calculates loss and update 'batch_loss' value in trainer state.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keyword arguments.
        """
        pred_batch = trainer.batch_state.pred
        y_batch = trainer.batch_state.y
        self._optimizer_step(pred_batch, y_batch, trainer)

    def on_optimization_end(self, trainer, **kwargs):
        """
        Sets the gradients of all optimized Tensors to zero.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keyword arguments.
        """
        self.optimizer.zero_grad()

    def on_validation_batch_end(self, trainer, **kwargs):
        """
        Calculates validation loss and update 'batch_loss' value in trainer state.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keyword arguments.
        """
        pred_batch = trainer.batch_state.pred
        y_batch = trainer.batch_state.y
        self._compute_loss(pred_batch, y_batch, trainer)
