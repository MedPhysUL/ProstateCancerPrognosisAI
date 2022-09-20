"""
    @file:              encoders.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     09/2022
    @Last modification: 09/2022

    @Description:       This file contains an abstract class named Encoder.
"""

from abc import ABC
from typing import List, Union

from torch import cat, nn, tensor

from src.models.base.blocks.base_block import BaseBlock


class Encoder(ABC):
    """
    Abstract class for encoder.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int
    ):
        """
        Saves the input size and the output size.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Number of features in the encodings.
        """
        self._input_size = input_size
        self._output_size = output_size

    @property
    def output_size(self):
        return self._output_size


class MLPEncodingBlock(Encoder, nn.Module):
    """
    An MLP encoding block is basically an MLP without prediction function.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
            layers: List[int],
            activation: str,
            dropout: float
    ):
        """
        Builds the layers of the encoding model.

        Parameters
        ----------
        input_size : int
            Number of features in the input.
        output_size : int
            Number of nodes in the last layer of the neural network.
        layers : List[int]
            List with number of units in each hidden layer.
        activation : str
            Name of the activation function.
        dropout : float
            Probability of dropout.
        """
        # Call of both parent constructors
        Encoder.__init__(self, input_size=input_size, output_size=output_size)
        nn.Module.__init__(self)

        # We create the layers
        layers.insert(0, input_size)
        layers.append(output_size)
        self.__layers = nn.Sequential(
            *[
                BaseBlock(
                    input_size=layers[i - 1],
                    output_size=layers[i],
                    activation=activation,
                    dropout=dropout
                ) for i in range(1, len(layers))
            ]
        )

    def forward(
            self,
            x: tensor
    ) -> tensor:
        """
        Executes the forward pass.

        Parameters
        ----------
        x : tensor
            (N,D) tensor with D-dimensional samples.

        Returns
        -------
        (N, D') tensor with concatenated embedding
        """
        return self.__layers(x)


class EntityEmbeddingBlock(nn.Module):
    """
    Contains a list of entity embedding layers associated to different categorical features.
    """

    def __init__(
            self,
            cat_sizes: Union[int, List[int]],
            cat_emb_sizes: Union[int, List[int]],
            cat_idx: List[int],
            embedding_sharing: bool = False
    ):
        """
        Creates a ModuleList with the embedding layers.

        Parameters
        ----------
        cat_sizes : Union[int, List[int]]
            List of integer representing the size of each categorical column.
        cat_emb_sizes : Union[int, List[int]]
            List of integer representing the size of each categorical embedding.
        cat_idx : List[int]
            List of idx associated to categorical columns in the dataset.
        """
        # Call of parent's constructor
        super().__init__()

        # We make sure the inputs are valid
        if len(cat_sizes) != len(cat_emb_sizes):
            raise ValueError('cat_sizes and cat_emb_sizes must be the same length')

        # We save the idx of categorical columns
        self.__cat_idx = cat_idx

        if embedding_sharing:

            # There is a single entity embedding layer for all columns in cat_idx
            emb_size = max(cat_emb_sizes)
            self.__output_size = emb_size*len(cat_idx)
            self.__generate_emb = self.__generate_shared_emb
            self.__embedding_layer = nn.Embedding(
                num_embeddings=max(cat_sizes),
                embedding_dim=emb_size
            )

        else:

            # We make another input validation
            if len(cat_idx) != len(cat_sizes):
                raise ValueError('cat_idx, cat_sizes and cat_emb_sizes must be all of the same'
                                 'length when embedding sharing is disabled')

            # There are separated embedding layers for each column in cat_idx
            self.__output_size = sum(cat_emb_sizes)
            self.__generate_emb = self.__generate_separated_emb
            self.__embedding_layer = nn.ModuleList(
                [nn.Embedding(cat_size, emb_size) for cat_size, emb_size in zip(cat_sizes, cat_emb_sizes)]
            )

    @property
    def output_size(self):
        return self.__output_size

    def __len__(self):
        """
        Returns the length of all embeddings concatenated
        """
        return self.output_size

    def __generate_separated_emb(self, x: tensor) -> tensor:
        """
        Generates the embeddings using the separated entity embedding layers and concatenate them.

        Parameters
        ----------
        x : tensor
            (N, C) tensor with C-dimensional samples where C is the number of categorical columns.

        Returns
        -------
        (N, output_size) tensor with concatenated embedding
        """
        embeddings = [e(x[:, i].long()) for i, e in enumerate(self.__embedding_layer)]
        return cat(embeddings, 1)

    def __generate_shared_emb(self, x: tensor) -> tensor:
        """
        Generates all the embeddings at once using a single shared entity embedding layer.

        Parameters
        ----------
        x : tensor
            (N, C) tensor with C-dimensional samples where C is the number of categorical columns.

        Returns
        -------
        (N, output_size) tensor with concatenated embedding
        """
        return self.__embedding_layer(x.long()).reshape(x.shape[0], self.__output_size)

    def forward(self, x: tensor) -> tensor:
        """
        Executes the forward pass.

        Parameters
        ----------
        x : tensor
            (N,D) tensor with D-dimensional samples

        Returns
        -------
        (N, D') tensor with concatenated embedding
        """
        return self.__generate_emb(x[:, self.__cat_idx])
