from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union

from monai.networks.nets import FullyConnectedNet
from torch import device as torch_device

from ..base import check_if_built, TorchModel
from ....data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..extraction import CNN


class RadiomicsMLP(TorchModel):

    def __init__(
            self,
            cnn: CNN,
            hidden_channels: Sequence[int],
            activation: Optional[Union[Tuple, str]] = None,
            adn_ordering: Optional[str] = None,
            bias: bool = True,
            dropout: Optional[Union[Tuple, str, float]] = None,
            device: Optional[torch_device] = None,
            name: Optional[str] = None,
            seed: Optional[int] = None
    ):
        super().__init__(device=device, name=name, seed=seed)

        self.cnn = cnn
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.adn_ordering = adn_ordering
        self.bias = bias
        self.dropout = dropout

        self.mlp = None

    def build(self, dataset: ProstateCancerDataset) -> RadiomicsMLP:
        """
        Builds the model using information contained in the dataset with which the model is going to be trained.

        Parameters
        ----------
        dataset : ProstateCancerDataset
            A prostate cancer dataset.

        Returns
        -------
        model : MLP
            The current model.
        """
        super().build(dataset=dataset)

        self.cnn.build(dataset)

        self.mlp = FullyConnectedNet(
            in_channels=self.cnn.n_features,
            out_channels=len(dataset.table_dataset.tasks),
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
            act=self.activation,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        ).to(self.device)

        return self

    @check_if_built
    def forward(
            self,
            features: FeaturesType
    ) -> TargetsType:
        """
        Executes the forward pass.

        Parameters
        ----------
        features : FeaturesType
            Batch data items.

        Returns
        -------
        predictions : TargetsType
            Predictions.
        """
        x_image = self.cnn.input_getter(features)
        radiomics = self.cnn.extractor(x_image)
        y = self.mlp(radiomics)
        y = {task.name: y[:, i] for i, task in enumerate(self._tasks.table_tasks)}

        return y
