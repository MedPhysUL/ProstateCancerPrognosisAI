"""
    @file:              sequential_net.py
    @Author:            Maxence Larose, Raphael Brodeur

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       This file is used to define a 'SequentialNet' model.
"""

from __future__ import annotations
from ast import literal_eval
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from torch import device as torch_device
from torch import cat, stack, sum, Tensor
from torch.nn import DataParallel, Module, ModuleDict

from .base import MultiTaskMode, Predictor
from ..blocks import BayesianFullyConnectedNet, FullyConnectedNet
from ....data.datasets.prostate_cancer import ProstateCancerDataset


class _Block(NamedTuple):
    """
    A block of the SequentialNet model.

    Elements
    --------
    input_task_names : Sequence[str]
        The input task names.
    task_name : str
        The task name.
    """
    input_task_names: Sequence[str]
    task_name: str


class SequentialNet(Predictor):
    """
    A SequentialNet model. The model can also be used to perform table tasks.
    """

    _ADDITIONAL_INPUTS_PER_BLOCKS = {0: [], 1: [0], 2: [0], 3: [0, 1], 4: [0, 1, 3], 5: [0, 1, 2, 3, 4]}

    def __init__(
            self,
            sequence: Sequence[str],
            n_layers: Union[int, Mapping[str, int]],
            n_neurons: Union[int, Mapping[str, int]],
            features_columns: Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]] = None,
            dropout: Union[float, Mapping[str, float]] = 0.0,
            activation: Union[Tuple, str] = "PRELU",
            bias: bool = True,
            adn_ordering: str = "NDA",
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None,
            bayesian: bool = False,
            temperature: Optional[Dict[str, float]] = None,
            prior_mean: float = 0.0,
            prior_variance: float = 0.1,
            posterior_mu_init: float = 0.0,
            posterior_rho_init: float = -3.0,
            standard_deviation: float = 0.1
    ):
        """
        Initializes the model.

        Parameters
        ----------
        sequence : Dict[str, int]
            A sequence of the task names.
        n_layers : Union[int, Mapping[str, int]]
            The number of layers per task. If a mapping is provided, the keys must be the task names.
        n_neurons : Union[int, Mapping[str, int]]
            The number of neurons per layer. If a mapping is provided, the keys must be the task names.
        features_columns : Optional[Union[str, Sequence[str], Mapping[str, Sequence[str]]]]
            The names of the features columns. If a mapping is provided, the keys must be the task names.
        dropout : Union[float, Mapping[str, float]]
            Probability of dropout. If a mapping is provided, the keys must be the task names. Defaults to 0.0.
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
        temperature : Optional[Dict[str, float]]
            Dictionary containing the temperature for each tasks. The temperature is the coefficient by which the KL
            divergence is multiplied when the loss is being computed. Keys are the task names and values are the
            temperature for each task.
        prior_mean : float
            Mean of the prior arbitrary Gaussian distribution to be used to calculate the KL divergence.
        prior_variance : float
            Prior variance used to calculate KL divergence.
        posterior_mu_init : float
            Initial value of the trainable mu parameter representing the mean of the Gaussian approximate of the
            posterior distribution.
        posterior_rho_init : float
            Rho parameter for reparametrization for the initial posterior distribution.
        standard_deviation : float
            Standard deviation of the gaussian distribution used to sample the initial posterior mu and initial
            posterior rho for the gaussian distribution from which the initial weights are sampled.
        """
        super().__init__(
            features_columns=features_columns,
            multi_task_mode=MultiTaskMode.SEPARATED,
            device=device,
            name=name,
            seed=seed,
            bayesian=bayesian,
            temperature=temperature
        )

        self.sequence = sequence

        if isinstance(n_layers, Mapping):
            if isinstance(n_neurons, Mapping):
                assert set(n_layers.keys()) == set(n_neurons.keys())
                self.hidden_channels = {t: [n_neurons[t]]*l for t, l in n_layers.items()}
            else:
                self.hidden_channels = {t: [n_neurons]*l for t, l in n_layers.items()}
        else:
            if isinstance(n_neurons, Mapping):
                self.hidden_channels = {t: [n]*n_layers for t, n in n_neurons.items()}
            else:
                self.hidden_channels = [n_neurons]*n_layers

        self.activation = activation
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.standard_deviation = standard_deviation

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
        for idx, task_name in enumerate(self.sequence):
            self._blocks.append(
                _Block(
                    input_task_names=[self.sequence[i] for i in self._ADDITIONAL_INPUTS_PER_BLOCKS[idx]],
                    task_name=task_name
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
                adn_ordering=self.adn_ordering,
                prior_mean=self.prior_mean,
                prior_variance=self.prior_variance,
                posterior_mu_init=self.posterior_mu_init,
                posterior_rho_init=self.posterior_rho_init,
                standard_deviation=self.standard_deviation
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

        return DataParallel(fully_connected_net).to(self.device)

    def _get_dropout(self, task_name: str) -> float:
        """
        Returns the dropout.

        Parameters
        ----------
        task_name : str
            The task name.

        Returns
        -------
        dropout : float
            The dropout.
        """
        if isinstance(self.dropout, Mapping):
            return self.dropout[task_name]
        else:
            return self.dropout

    def _get_hidden_channels(self, task_name: str) -> Sequence[int]:
        """
        Returns the hidden channels.

        Parameters
        ----------
        task_name : str
            The task name.

        Returns
        -------
        hidden_channels : Sequence[int]
            The hidden channels.
        """
        if isinstance(self.hidden_channels, Mapping):
            return self.hidden_channels[task_name]
        else:
            return self.hidden_channels

    def _get_in_channels(self, task_name: str) -> int:
        """
        Returns the number of input channels.

        Parameters
        ----------
        task_name : str
            The task name.

        Returns
        -------
        in_channels : int
            The number of input channels.
        """
        if isinstance(self.features_columns, Mapping):
            base_length = len(self.features_columns[task_name])
        else:
            base_length = len(self.features_columns)

        return base_length + len(self._ADDITIONAL_INPUTS_PER_BLOCKS[self.sequence.index(task_name)])

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
                dropout=self._get_dropout(task.name),
                hidden_channels=self._get_hidden_channels(task.name),
                in_channels=self._get_in_channels(task.name),
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
        base_input = table_data[block.task_name] if isinstance(table_data, dict) else table_data
        additional_inputs = [output[t] for t in block.input_task_names]

        if additional_inputs:
            predictor_input = cat([base_input, stack(additional_inputs, 1)], 1)
        else:
            predictor_input = base_input

        return predictor_input

    def _get_kl_divergence(
            self,
            base_kl: Tensor,
            block: _Block,
            kl_divergence: Dict[str, Tensor]
    ) -> Tensor:
        """
        Returns the KL divergence.

        Parameters
        ----------
        base_kl : Tensor
            The base KL divergence.
        block : _Block
            The current SequentialNet block.
        kl_divergence : Dict[str, Tensor]
            Previous prediction iteration KL divergences.

        Returns
        -------
        kl_divergence : Tensor
            The KL divergence.
        """
        return sum(stack([base_kl, *[kl_divergence[t] for t in block.input_task_names]]))

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
                predictor_input = self.__get_predictor_input(table_data=table_data, block=block, output=output)

                y, kl = self.predictor[block.task_name](predictor_input)
                output[block.task_name] = y[:, 0]
                kl_divergence[block.task_name] = self._get_kl_divergence(kl, block, kl_divergence)

            return output, kl_divergence

        else:
            for block in self._blocks:
                predictor_input = self.__get_predictor_input(table_data=table_data, block=block, output=output)

                output[block.task_name] = self.predictor[block.task_name](predictor_input)[:, 0]

            return output
