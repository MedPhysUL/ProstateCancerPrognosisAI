"""
    @file:              bt_blocks.py
    @Author:            Maxence Larose

    @Creation Date:     07/2023
    @Last modification: 07/2023

    @Description:       A modified version of multiple blocks from the bayesian_torch library.
"""

from bayesian_torch.layers.variational_layers.conv_variational import (
    Conv3dReparameterization,
    ConvTranspose3dReparameterization
)
from bayesian_torch.layers.variational_layers.linear_variational import LinearReparameterization


class BayesianConv3D(Conv3dReparameterization):
    """
    Implements Conv3d layer with reparameterization trick. Inherits from Conv3dReparameterization.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            prior_mean: float,
            prior_variance: float,
            posterior_mu_init: float,
            posterior_rho_init: float,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            standard_deviation: float = 0.1
    ):
        """
        Implements Conv3d layer with reparameterization trick. Inherits from Conv3dReparameterization.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int
            Size of the convolving kernel.
        prior_mean : float
            Mean of the prior arbitrary distribution to be used on the complexity cost.
        prior_variance : float
            Variance of the prior arbitrary distribution to be used on the complexity cost.
        posterior_mu_init : float
            Init trainable mu parameter representing mean of the approximate posterior.
        posterior_rho_init : float
            Init trainable rho parameter representing the sigma of the approximate posterior through softplus function.
        stride : int
            Stride of the convolution. Default: 1.
        padding : int
            Zero-padding added to both sides of the input. Default: 0.
        dilation : int
            Spacing between kernel elements. Default: 1.
        groups : int
            Number of blocked connections from input channels to output channels.
        bias : bool
            If set to False, the layer will not learn an additive bias. Default: True.
        standard_deviation : float
            Standard deviation of the gaussian distribution used to initialize the weights.
        """
        self.standard_deviation = standard_deviation

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def init_parameters(self):
        """
        Initializes the parameters of the layer.
        """
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)

        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)


class BayesianConvTranspose3D(ConvTranspose3dReparameterization):
    """
    Implements ConvTranspose3d layer with reparameterization trick. Inherits from ConvTranspose3dReparameterization.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            output_padding=0,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0,
            bias=True,
            standard_deviation: float = 0.1
    ):
        """
        Implements ConvTranspose3d layer with reparameterization trick. Inherits from ConvTranspose3dReparameterization.

        Parameters
        ----------
        in_channels : int
            Number of channels in the input image.
        out_channels : int
            Number of channels produced by the convolution.
        kernel_size : int
            Size of the convolving kernel.
        stride : int
            Stride of the convolution. Default: 1.
        padding : int
            Zero-padding added to both sides of the input. Default: 0.
        dilation : int
            Spacing between kernel elements. Default: 1.
        groups : int
            Number of blocked connections from input channels to output channels.
        output_padding : int
            Additional size added to one side of each dimension in the output shape. Default: 0.
        prior_mean : float
            Mean of the prior arbitrary distribution to be used on the complexity cost.
        prior_variance : float
            Variance of the prior arbitrary distribution to be used on the complexity cost.
        posterior_mu_init : float
            Init trainable mu parameter representing mean of the approximate posterior.
        posterior_rho_init : float
            Init trainable rho parameter representing the sigma of the approximate posterior through softplus function.
        bias : bool
            If set to False, the layer will not learn an additive bias. Default: True.
        standard_deviation : float
            Standard deviation of the gaussian distribution used to initialize the weights.
        """
        self.standard_deviation = standard_deviation

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def init_parameters(self):
        """
        Initializes the parameters of the layer.
        """
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_kernel.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
        self.rho_kernel.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)
        if self.bias:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)

            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)


class BayesianLinear(LinearReparameterization):
    """
    Implements Linear layer with reparameterization trick. Inherits from LinearReparameterization.
    """

    def __init__(
            self,
            in_features,
            out_features,
            prior_mean=0,
            prior_variance=1,
            posterior_mu_init=0,
            posterior_rho_init=-3.0,
            bias=True,
            standard_deviation: float = 0.1
    ):
        """
        Implements Linear layer with reparameterization trick. Inherits from LinearReparameterization.

        Parameters:
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        prior_mean : float
            Mean of the prior arbitrary distribution to be used on the complexity cost,
        prior_variance : float
            Variance of the prior arbitrary distribution to be used on the complexity cost.
        posterior_mu_init : float
            Init trainable mu parameter representing mean of the approximate posterior.
        posterior_rho_init : float
            Init trainable rho parameter representing the sigma of the approximate posterior through softplus function.
        bias: bool
            If set to False, the layer will not learn an additive bias. Default: True.
        standard_deviation : float
            Standard deviation of the gaussian distribution used to initialize the weights.
        """
        self.standard_deviation = standard_deviation

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def init_parameters(self):
        """
        Initializes the parameters of the layer.
        """
        self.prior_weight_mu.fill_(self.prior_mean)
        self.prior_weight_sigma.fill_(self.prior_variance)

        self.mu_weight.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
        self.rho_weight.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)
        if self.mu_bias is not None:
            self.prior_bias_mu.fill_(self.prior_mean)
            self.prior_bias_sigma.fill_(self.prior_variance)
            self.mu_bias.data.normal_(mean=self.posterior_mu_init[0], std=self.standard_deviation)
            self.rho_bias.data.normal_(mean=self.posterior_rho_init[0], std=self.standard_deviation)
