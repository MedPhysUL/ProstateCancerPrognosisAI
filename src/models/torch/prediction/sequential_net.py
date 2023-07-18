"""
    @file:              sequential_net.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file is used to define a 'SequentialNet' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from torch import device as torch_device
from torch import cat, stack, Tensor
from torch.nn import DataParallel, Module, ModuleDict

from .base import MultiTaskMode, Predictor
from ..blocks import BayesianFullyConnectedNet, FullyConnectedNet
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _Block(NamedTuple):
    """
    A block of the SequentialNet model.

    Elements
    --------
    input_target_columns : Sequence[str]
        The input target columns.
    target_column : str
        The target column.
    """
    input_target_columns: Sequence[str]
    target_column: str


class SequentialNet(Predictor):
    """
    A SequentialNet model. The model can also be used to perform table tasks.
    """

    _ADDITIONAL_INPUTS_PER_BLOCKS = {0: [], 1: [0], 2: [0], 3: [1], 4: [3], 5: [2, 4]}

    def __init__(
            self,
            sequence: Sequence[str],
            features_columns: Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
            hidden_channels: Union[str, Sequence[int], Mapping[str, Sequence[int]]] = (25, 25, 25),
            dropout: Union[float, Mapping[str, float]] = 0.0,
            activation: Union[Tuple, str] = "PRELU",
            bias: bool = True,
            adn_ordering: str = "NDA",
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
            bayesian: bool = False
    ):
        """
        Initializes the model.

        Parameters
        ----------
        sequence : Dict[str, int]
            A sequence of the target columns.
        features_columns : Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]]
            The names of the features columns. If a mapping is provided, the keys must be the target columns associated
            to the tasks.
        hidden_channels : Union[str, Sequence[int], Mapping[str, Sequence[int]]]
            List with number of units in each hidden layer. If a mapping is provided, the keys must be the target
            columns associated to the tasks. Defaults to (25, 25, 25).
        dropout : Union[float, Mapping[str, float]]
            Probability of dropout. If a mapping is provided, the keys must be the target columns associated to the
            tasks. Defaults to 0.0.
        activation : Union[Tuple, str]
            Activation function. Defaults to "PRELU".
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
        bayesian : bool
            Whether the model should implement variational inference.
        """
        super().__init__(
            features_columns=features_columns,
            multi_task_mode=MultiTaskMode.SEPARATED,
            device=device,
            name=name,
            seed=seed,
            bayesian=bayesian
        )

        self.sequence = sequence

        if isinstance(hidden_channels, Mapping):
            self.hidden_channels = {t: literal_eval(c) if isinstance(c, str) else c for t, c in hidden_channels.items()}
        elif isinstance(hidden_channels, str):
            self.hidden_channels = literal_eval(hidden_channels)
        else:
            self.hidden_channels = hidden_channels

        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self._blocks: Optional[List[_Block]] = None
        self.predictor = None

    def _validate_sequence(self):
        """
        Validates the sequence.
        """
        assert len(self.sequence) == len(self._tasks.table_tasks), (
            f"The number of target columns in the sequence must be equal to the number of tasks. Expected "
            f"{len(self._tasks.table_tasks)} but got {len(self.sequence)}."
        )

    def _create_blocks(self):
        """
        Creates the blocks of the sequence.
        """
        self._blocks = []
        for idx, target_column in enumerate(self.sequence):
            self._blocks.append(
                _Block(
                    input_target_columns=[self.sequence[i] for i in self._ADDITIONAL_INPUTS_PER_BLOCKS[idx]],
                    target_column=target_column
                )
            )

    def _build_single_predictor(
            self,
            dropout: float,
            hidden_channels: Sequence[int],
            in_channels: int,
            out_channels: int
    ) -> Module:
        """
        Returns a single predictor module.

        Parameters
        ----------
        dropout : float
            Probability of dropout.
        hidden_channels : Sequence[int]
            List with number of units in each hidden layer.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.

        Returns
        -------
        predictor : Module
            A single predictor module.
        """
        if self.bayesian:
            fully_connected_net = BayesianFullyConnectedNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                act=self.activation,
                bias=self.bias,
                adn_ordering=self.adn_ordering
            )
        else:
            fully_connected_net = FullyConnectedNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                act=self.activation,
                bias=self.bias,
                adn_ordering=self.adn_ordering
            )
        fully_connected_net = DataParallel(fully_connected_net)

        return fully_connected_net.to(self.device)

    def _get_dropout(self, target_column: str) -> float:
        """
        Returns the dropout.

        Parameters
        ----------
        target_column : str
            The target column.

        Returns
        -------
        dropout : float
            The dropout.
        """
        if isinstance(self.dropout, Mapping):
            return self.dropout[target_column]
        else:
            return self.dropout

    def _get_hidden_channels(self, target_column: str) -> Sequence[int]:
        """
        Returns the hidden channels.

        Parameters
        ----------
        target_column : str
            The target column.

        Returns
        -------
        hidden_channels : Sequence[int]
            The hidden channels.
        """
        if isinstance(self.hidden_channels, Mapping):
            return self.hidden_channels[target_column]
        else:
            return self.hidden_channels

    def _get_in_channels(self, target_column: str) -> int:
        """
        Returns the number of input channels.

        Parameters
        ----------
        target_column : str
            The target column.

        Returns
        -------
        in_channels : int
            The number of input channels.
        """
        if isinstance(self.features_columns, Mapping):
            base_length = len(self.features_columns[target_column])
        else:
            base_length = len(self.features_columns)

        return base_length + len(self._ADDITIONAL_INPUTS_PER_BLOCKS[self.sequence.index(target_column)])

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
        self._validate_sequence()
        self._create_blocks()

        predictor = ModuleDict()
        for task in self._tasks.table_tasks:
            predictor[task.name] = self._build_single_predictor(
                dropout=self._get_dropout(task.target_column),
                hidden_channels=self._get_hidden_channels(task.target_column),
                in_channels=self._get_in_channels(task.target_column),
                out_channels=1
            )

        return predictor

    def __get_predictor_input(
            self,
            table_data: Union[Tensor, Dict[str, Tensor]],
            output: Dict[str, Tensor],
            block: _Block
    ):
        """
        Gets the input to the predictor.

        Parameters
        ----------
        table_data : Union[Tensor, Dict[str, Tensor]]
            The table data.
        output : Dict[str, Tensor]
            Previous prediction iteration outputs.
        block : _Block
            The current SequentialNet block.

        Returns
        -------
        predictor_input
            The input to the predictor.
        """
        task_name = self.map_from_target_col_to_task_name[block.target_column]

        base_input = table_data[task_name] if isinstance(table_data, dict) else table_data
        additional_inputs = [output[self.map_from_target_col_to_task_name[c]] for c in block.input_target_columns]

        if additional_inputs:
            predictor_input = cat([base_input, stack(additional_inputs, 1)], 1)
        else:
            predictor_input = base_input

        return predictor_input

    def _get_prediction(self, table_data: Union[Tensor, Dict[str, Tensor]]) -> Union[Dict[str, Tensor], tuple]:
        """
        Returns the prediction.

        Parameters
        ----------
        table_data : Union[Tensor, Dict[str, Tensor]]
            The table data.

        Returns
        -------
        prediction : Union[Dict[str, Tensor], tuple]
            The prediction and its KL divergence (if the model is in bayesian mode).
        """
        output = {}

        if self.bayesian:
            kl_divergence = {}
            for block in self._blocks:
                task_name = self.map_from_target_col_to_task_name[block.target_column]

                predictor_input = self.__get_predictor_input(table_data=table_data, block=block, output=output)

                y, kl = self.predictor[task_name](predictor_input)
                output[task_name] = y[:, 0]
                kl_divergence[task_name] = kl

            return output, kl_divergence

        else:
            for block in self._blocks:
                task_name = self.map_from_target_col_to_task_name[block.target_column]

                predictor_input = self.__get_predictor_input(table_data=table_data, block=block, output=output)

                output[task_name] = self.predictor[task_name](predictor_input)[:, 0]

            return output
