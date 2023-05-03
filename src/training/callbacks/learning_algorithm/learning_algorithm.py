"""
    @file:              learning_algorithm.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define a 'LearningAlgorithm' which dictates how to update the model's
                        parameters during the training process.
"""

from copy import copy
from itertools import count
from typing import Dict, Iterable, Optional, Union

from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..base import Priority, TrainingCallback
from .early_stopper import EarlyStopper
from ....losses.multi_task.base import MultiTaskLoss
from .regularizer import Regularizer, RegularizerList


class LearningAlgorithm(TrainingCallback):
    """
    This class is used to dictate how to update the model's parameters during the training process.
    """

    instance_counter = count()

    CLASS_NAME_KEY = "class_name"

    TORCH_LIKE_SERIALIZABLE_ATTRIBUTES = ["criterion", "optimizer", "lr_scheduler", "regularizer"]
    PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES = ["early_stopper"]

    def __init__(
            self,
            criterion: MultiTaskLoss,
            optimizer: Optimizer,
            early_stopper: Optional[EarlyStopper] = None,
            lr_scheduler: Optional[LRScheduler] = None,
            name: Optional[str] = None,
            regularizer: Optional[Union[Regularizer, RegularizerList, Iterable[Regularizer]]] = None,
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
        early_stopper : Optional[EarlyStopper]
            An early stopper.
        lr_scheduler : Optional[LRScheduler]
            A pytorch learning rate scheduler.
        name : Optional[str]
            The name of the callback.
        regularizer : Optional[Union[Regularizer, RegularizerList, Iterable[Regularizer]]]
            Regularizer.
        """
        self.instance_id = next(self.instance_counter)
        name = name if name else f"{self.__class__.__name__}({self.instance_id})"
        super().__init__(name=name, **kwargs)

        self.clip_grad_max_norm = kwargs.get("clip_grad_max_norm", 3.0)
        self.criterion = criterion
        self.early_stopper = early_stopper
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.regularizer = RegularizerList(regularizer)
        self.stopped = False

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.MEDIUM_PRIORITY

    @property
    def allow_duplicates(self) -> bool:
        """
        Whether to allow duplicates of this specific Callback class in the 'TrainingCallbackList'.

        Returns
        -------
        allow : bool
            Allow duplicates.
        """
        return True

    def state_dict(self) -> Optional[dict]:
        """
        Gets the state of the callback.

        Returns
        -------
        state : Optional[dict]
            The state of the callback.
        """
        if self.save_state:
            state = {}
            for k, v in vars(self).items():
                if k not in self.UNSERIALIZABLE_ATTRIBUTES:
                    if k in self.TORCH_LIKE_SERIALIZABLE_ATTRIBUTES:
                        if v:
                            attribute_state = v.state_dict()
                            attribute_state[self.CLASS_NAME_KEY] = v.__class__.__name__
                            state[k] = attribute_state
                        else:
                            state[k] = None
                    elif k in self.PYTHON_LIKE_SERIALIZABLE_ATTRIBUTES:
                        if v:
                            attribute_state = vars(v).copy()
                            attribute_state[self.CLASS_NAME_KEY] = v.__class__.__name__
                            state[k] = attribute_state
                        else:
                            state[k] = None
                    else:
                        state[k] = copy(v)

            return state

        return None

    def _update_batch_state(
            self,
            single_task_losses: Dict[str, Tensor],
            multi_task_loss_without_regularization: Tensor,
            multi_task_loss_with_regularization: Optional[Tensor],
            trainer
    ):
        """
        Updates current batch state.

        Parameters
        ----------
        single_task_losses : Dict[str, Tensor]
            A dictionary containing single task losses. Keys are loss names and values are loss values.
        multi_task_loss_without_regularization : Tensor
            The multi-task loss, excluding the regularization term (penalty).
        multi_task_loss_with_regularization : Optional[Tensor]
            The multi-task loss, including the regularization term (penalty).
        trainer : Trainer
            The trainer.
        """
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
        Calls the criterion and add the elastic penalty to compute total loss.

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
        if self.regularizer:
            regularization = self.regularizer(device=loss_without_regularization.device)
            loss_with_regularization = loss_without_regularization + regularization

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
        if not self.criterion.tasks:
            self.criterion.tasks = trainer.training_state.tasks

        if self.early_stopper:
            self.early_stopper.on_fit_start(self, trainer)

    def _optimizer_step(self, pred_batch: Dict[str, Tensor], y_batch: Dict[str, Tensor], trainer):
        """
        Performs an optimizer step, i.e computes loss and performs backward pass.

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
        batch_loss.backward(retain_graph=True)
        clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.clip_grad_max_norm)
        self.optimizer.step()

    def on_optimization_start(self, trainer, **kwargs):
        """
        Computes loss and updates 'batch_loss' value in trainer state.

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

    def on_epoch_end(self, trainer, **kwargs):
        """
        Performs a learning rate scheduler step and checks if early stop needs to occur.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        kwargs : dict
            Keyword arguments.
        """
        if self.lr_scheduler:
            self.lr_scheduler.step()

        if self.early_stopper:
            if self.early_stopper(trainer):
                self.stopped = True

    def on_fit_end(self, trainer, **kwargs):
        """
        Called when the training is finished. If the last model is not the best model, loads the best model.

        Parameters
        ----------
        trainer : Trainer
            The trainer.
        """
        if self.early_stopper:
            self.early_stopper.load_best_model(trainer.model)
            self.early_stopper.set_best_epoch(trainer)
