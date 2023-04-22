"""
    @file:              mlp.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file is used to define an 'MLP' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Optional, Sequence, Tuple, Union

from monai.networks.nets import FullyConnectedNet
from torch import device as torch_device
from torch.nn import Module, ModuleDict

from .base import InputMode, MultiTaskMode, Predictor
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class MLP(Predictor):
    """
    A simple MLP model. The model can take as input either the tabular features, the radiomics features or both. The
    model can also be used to perform table tasks on the extracted deep radiomics.
    """

    def __init__(
            self,
            features_columns: Optional[Union[str, Sequence[str]]] = None,
            input_mode: Union[str, InputMode] = InputMode.TABULAR,
            multi_task_mode: Union[str, MultiTaskMode] = MultiTaskMode.FULLY_SHARED,
            n_radiomics: int = 5,
            hidden_channels: Union[str, Sequence[int]] = (25, 25, 25),
            activation: Union[Tuple, str] = "PRELU",
            dropout: Union[Tuple, str, float] = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        """
        Initializes the model.

        Parameters
        ----------
        features_columns : Optional[Union[str, Sequence[str]]]
            The names of the features columns. This parameter is only used when `input_mode` is set to 'tabular' or
            'hybrid'.
        input_mode : Union[str, InputMode]
            The input mode of the model. Available modes are 'tabular', 'radiomics' and 'hybrid'. Defaults to 'tabular'.
        multi_task_mode : Union[str, MultiTaskMode]
            Available modes are 'separated' or 'fully_shared'. If 'separated', a separate extractor model is used for
            each task. If 'fully_shared', a fully shared extractor model is used. All layers are shared between the
            tasks.
        n_radiomics : int
            Number of radiomics features. Defaults to 5. This parameter is only used when `input_mode` is set to
            'radiomics' or 'hybrid'.
        hidden_channels : Union[str, Sequence[int]]
            List with number of units in each hidden layer. Defaults to (25, 25, 25).
        activation : Union[Tuple, str]
            Activation function. Defaults to "PRELU".
        dropout : Union[Tuple, str, float]
            Probability of dropout. Defaults to 0.0.
        bias : bool
            If `bias` is True then linear units have a bias term. Defaults to True.
        adn_ordering : str
            A string representing the ordering of activation, dropout, and normalization. Defaults to "NDA".
        device : Optional[torch_device]
            The device of the model. Defaults to None.
        name : Optional[str]
            The name of the model. Defaults to None.
        seed : Optional[int]
            Random state used for reproducibility. Defaults to None.
        """
        super().__init__(
            features_columns=features_columns,
            input_mode=input_mode,
            multi_task_mode=multi_task_mode,
            n_radiomics=n_radiomics,
            device=device,
            name=name,
            seed=seed
        )

        self.hidden_channels = literal_eval(hidden_channels) if isinstance(hidden_channels, str) else hidden_channels
        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.predictor = None

    def _build_single_predictor(self, out_channels: int) -> Module:
        """
        Returns a single predictor module.

        Parameters
        ----------
        out_channels : int
            Number of output channels.

        Returns
        -------
        predictor : Module
            A single predictor module.
        """
        if self.input_mode == InputMode.RADIOMICS:
            input_size = self.n_radiomics
        else:
            table_input_size = len(self.features_columns)
            if self.input_mode == InputMode.TABULAR:
                input_size = table_input_size
            elif self.input_mode == InputMode.HYBRID:
                input_size = table_input_size + self.n_radiomics
            else:
                raise ValueError(f"Invalid input_mode: {self.input_mode}")

        return FullyConnectedNet(
            in_channels=input_size,
            out_channels=out_channels,
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            act=self.activation,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        ).to(self.device)

    def _build_predictor(self, dataset: ProstateCancerDataset) -> Union[Module, ModuleDict]:
        """
        Builds the model. This method must be called before training the model.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : Union[Module, ModuleDict]
            The current model.
        """
        if self.multi_task_mode == MultiTaskMode.SEPARATED:
            return ModuleDict(
                {task.name: self._build_single_predictor(out_channels=1) for task in self._tasks.table_tasks}
            )
        elif self.multi_task_mode == MultiTaskMode.FULLY_SHARED:
            return self._build_single_predictor(out_channels=len(self._tasks.table_tasks))
