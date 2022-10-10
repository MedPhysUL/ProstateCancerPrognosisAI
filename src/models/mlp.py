"""
    @file:              mlp.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file is used to define the regression and classification wrappers for MLP models.
"""

from typing import List, Optional

from src.models.base.torch_wrapper import TorchWrapper
from src.models.base.mlp_base_model import MLPBaseModel
from src.training.early_stopper import EarlyStopperType
from src.utils.hyperparameters import CategoricalHP, HP, NumericalContinuousHP, NumericalIntHP
from src.utils.multi_task_losses import MultiTaskLoss


class MLPHP:
    """
    MLP hyperparameters
    """
    ACTIVATION = CategoricalHP("activation")
    ALPHA = NumericalContinuousHP("alpha")
    BATCH_SIZE = NumericalIntHP("batch_size")
    BETA = NumericalContinuousHP("beta")
    DROPOUT = NumericalContinuousHP("dropout")
    LR = NumericalContinuousHP("lr")
    N_LAYER = NumericalIntHP("n_layer")
    N_UNIT = NumericalIntHP("n_unit")
    RHO = NumericalContinuousHP("rho")

    def __iter__(self):
        return iter(
            [self.ACTIVATION, self.ALPHA, self.BATCH_SIZE, self.DROPOUT, self.BETA, self.LR, self.N_LAYER, self.N_UNIT,
             self.RHO]
        )


class MLP(TorchWrapper):
    """
    Multilayer perceptron classification model wrapper.
    """

    def __init__(
            self,
            output_size: int,
            path_to_model: str,
            criterion: MultiTaskLoss,
            n_layer: int,
            n_unit: int,
            activation: str,
            early_stopper_type: EarlyStopperType,
            patience: int = 10,
            dropout: float = 0,
            alpha: float = 0,
            beta: float = 0,
            lr: float = 0.05,
            rho: float = 0,
            batch_size: int = 55,
            valid_batch_size: Optional[int] = None,
            max_epochs: int = 200,
            verbose: bool = False,
    ):
        """
        Builds a binary classification MLP and sets the protected attributes using parent's constructor.

        Parameters
        ----------
        output_size : int
            Number of nodes in the last layer of the neural network
        path_to_model : str
            Path to save model.
        criterion : MultiTaskLoss
            Loss function of our model
        n_layer : int
            Number of hidden layer
        n_unit : int
            Number of units in each hidden layer
        activation : str
            Activation function
        early_stopper_type : EarlyStopperType
            Early stopper type.
        patience : int
            Patience.
        dropout : float
            Probability of dropout
        alpha : float
            L1 penalty coefficient
        beta : float
            L2 penalty coefficient
        verbose : bool
            True if we want trace of the training progress
        lr : float
            Learning rate
        rho : float
            If >=0 will be used as neighborhood size in Sharpness-Aware Minimization optimizer, otherwise,
            Adam optimizer will be used
        batch_size : int
            Size of the batches in the training loader
        valid_batch_size : int
            Size of the batches in the valid loader (None = one single batch)
        max_epochs : int
            Maximum number of epochs for training
        verbose : bool
            If True, training progress will be printed
        """
        super().__init__(
            model_constructor=MLPBaseModel,
            model_params=dict(
                output_size=output_size,
                criterion=criterion,
                path_to_model=path_to_model,
                layers=[n_unit]*n_layer,
                activation=activation,
                dropout=dropout,
                alpha=alpha,
                beta=beta,
                verbose=verbose
            ),
            train_params=dict(
                lr=lr,
                rho=rho,
                early_stopper_type=early_stopper_type,
                patience=patience,
                batch_size=batch_size,
                valid_batch_size=valid_batch_size,
                max_epochs=max_epochs)
        )

    @staticmethod
    def get_hps() -> List[HP]:
        """
        Returns a list with the hyperparameters associated to the model
        Returns: list of hyperparameters
        """
        return list(MLPHP())
