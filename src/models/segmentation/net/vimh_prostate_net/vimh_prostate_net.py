"""
    @file:              vimh_prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 03/2023

    @Description:       This file contains the VIMHProstateNet class which is a bayesian segmentation model. The model
                        is a multi head variational inference version of ProstateNet.
"""

import torch
import torch.nn as nn

from src.models.segmentation.net.blocks.blocks import EncoderBlock, DecoderBlock, BayesianDecoderBlock


# TODO -- Currently, the heads are planted at a fix depth of dec3. Depth should be variable.


class DeterministicBody(nn.Module):
    """
    A class that contains the deterministic body to be used by VIMHProstateNet.
    """

    def __init__(
            self,
            channels: tuple[int, int, int, int, int],
            dropout: float = 0.2
    ):
        """
        Builds the layers of the deterministic body of VIMHProstateNet.

        Parameters
        ----------
        channels : tuple[int, int, int, int, int]
            Tuple of the amount of channels at each of the 5 layers of the full VIMH model.
        dropout : float
            Dropout probability.
        """
        super().__init__()

        self.enc1 = EncoderBlock(input_channels=1, output_channels=channels[0], dropout=dropout)
        self.enc2 = EncoderBlock(input_channels=channels[0], output_channels=channels[1], dropout=dropout)
        self.enc3 = EncoderBlock(input_channels=channels[1], output_channels=channels[2], dropout=dropout)
        self.enc4 = EncoderBlock(input_channels=channels[2], output_channels=channels[3], dropout=dropout)
        self.bottom = EncoderBlock(input_channels=channels[3], output_channels=channels[4], stride=1, dropout=dropout)
        self.dec4 = DecoderBlock(input_channels=channels[4] + channels[3], output_channels=channels[2], dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Defines the forward method of the body of VIMHProstateNet.

        Parameters
        ----------
        x : torch.Tensor
            Input of the model (a medical image).

        Returns
        -------
        return : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Input of the heads. Tuple of the output of the blocks of the body to be used by the heads.
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottom = self.bottom(enc4)
        dec4 = self.dec4(torch.cat([enc4, bottom], dim=1))

        return enc1, enc2, enc3, dec4


class BayesianHead(nn.Module):
    """
    A class that contains a variational inference head to be used by VIMHProstateNet.
    """

    def __init__(
            self,
            channels: tuple[int, int, int, int, int],
            dropout: float = 0.2
    ):
        """
        Builds the layers of a variational inference head of VIMHProstateNet.

        Parameters
        ----------
        channels : tuple[int, int, int, int, int]
            Tuple of the amount of channels at each of the 5 layers of the full VIMH model.
        dropout : float
            Dropout probability.
        """
        super().__init__()

        self.dec3 = BayesianDecoderBlock(input_channels=channels[2] * 2, output_channels=channels[1], dropout=dropout)
        self.dec2 = BayesianDecoderBlock(input_channels=channels[1] * 2, output_channels=channels[0], dropout=dropout)
        self.dec1 = BayesianDecoderBlock(input_channels=channels[0] * 2, output_channels=1, is_top=True, dropout=dropout)

    def forward(
            self,
            enc1: torch.Tensor,
            enc2: torch.Tensor,
            enc3: torch.Tensor,
            dec4: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """
        Defines the forward method of a head of VIMHProstateNet.

        Parameters
        ----------
        enc1 : torch.Tensor
            Output of the encoder block enc1 of the body.
        enc2 : torch.Tensor
            Output of the encoder block enc2 of the body.
        enc3 : torch.Tensor
            Output of the encoder block enc3 of the body.
        dec4 : torch.Tensor
            Output of the decoder block dec4 of the body.

        Returns
        -------
        return : tuple[torch.Tensor, float]
            Tuple containing the transformed tensor (i.e. the segmentation) and the sum of the KL divergence of each
            layer of the head.
        """
        kl_sum = 0

        dec3, kl = self.dec3(torch.cat([enc3, dec4], dim=1))
        kl_sum += kl

        dec2, kl = self.dec2(torch.cat([enc2, dec3], dim=1))
        kl_sum += kl

        dec1, kl = self.dec1(torch.cat([enc1, dec2], dim=1))
        kl_sum += kl

        return dec1, kl_sum


class VIMHProstateNet(nn.Module):
    """
    A class that contains a VIMH segmentation model. The model is based on a UNet architecture and incorporates
    residual paths. Multiple variational inference heads are connected on a shared deterministic body.
    """

    def __init__(
            self,
            num_heads: int,
            channels: tuple[int, int, int, int, int],
            body_dropout: float = 0.2,
            head_dropout: float = 0.2
    ):
        """
        Gets the body and the heads of the model.

        Parameters
        ----------
        num_heads : int
            Number of heads
        channels : tuple[int, int, int, int, int]
            Tuple of the amount of channels at each of the 5 layers of the model.
        body_dropout : float
            Dropout probability to be applied in the body of the model.
        head_dropout : float
            Dropout probability to be applied in the heads of the model.
        """
        super().__init__()

        self.num_heads = num_heads

        self.body = DeterministicBody(channels, dropout=body_dropout)
        self.heads = nn.ModuleList([BayesianHead(channels, dropout=head_dropout) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward method of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input of the model (a medical image).

        Returns
        -------
        return : tuple[torch.Tensor, torch.Tensor]
            Tensor containing the transformed tensor (i.e. the segmentation) and the sum of the KL divergence of each
            head.
        """
        enc1, enc2, enc3, dec4 = self.body(x)

        y_list = []
        kl_list = []
        for i in range(self.num_heads):
            y, kl = self.heads[i](enc1, enc2, enc3, dec4)

            y_list.append(y)
            kl_list.append(kl)

        return torch.stack(y_list), torch.stack(kl_list)
