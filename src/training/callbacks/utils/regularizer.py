"""
    @file:              regularizer.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define regularizers which are used to add some penalty to the loss
                        function.
"""

from abc import abstractmethod
from typing import Any, Dict, Iterable, Iterator, Optional, Union

from torch import device as torch_device
from torch import linalg, stack, sum, tensor, Tensor
from torch.nn import Module, Parameter, ParameterList


class Regularizer(Module):
    """
    Base class for regularizer.
    """

    LAMBDA = "lambda_"

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            name: Optional[str] = None
    ):
        """
        Constructor of the Regularizer class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularizer. In other words, the coefficient that multiplies the loss.
        name : Optional[str]
            The name of the regularizer.
        """
        super().__init__()

        if isinstance(params, dict):
            params = list(params.values())
        else:
            params = list(params)

        self.params = ParameterList(params)
        self.lambda_ = lambda_
        self.name = name if name else f"{self.__class__.__name__}('lambda_'={lambda_})"

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Call the forward pass of the regularizer and scale it by the 'lambda_' attribute.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The regularization loss.
        """
        out = super().__call__(*args, **kwargs)
        return self.lambda_ * out

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Compute the forward pass of the regularizer.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The regularization loss.
        """
        raise NotImplementedError

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Get the state of the regularizer.

        Returns
        -------
        state: Dict[str, Any]
            The state of the regularizer.
        """
        state_dict = super().state_dict(**kwargs)
        state_dict[self.LAMBDA] = self.lambda_
        return state_dict


class RegularizerList(Module):
    """
    Holds regularizers in a list.
    """

    def __init__(
            self,
            regularizers: Optional[Union[Regularizer, Iterable[Regularizer]]] = None
    ):
        """
        Constructor of the RegularizerList class.

        Parameters
        ----------
        regularizers : Optional[Union[Regularizer, Iterable[Regularizer]]]
            The regularizers to apply.
        """
        super().__init__()

        if regularizers is None:
            regularizers = []
        if isinstance(regularizers, Regularizer):
            regularizers = [regularizers]

        assert isinstance(regularizers, Iterable), "regularizers must be an Iterable."
        assert all(isinstance(regularizer, Regularizer) for regularizer in regularizers), (
            "All regularizer must be instances of Regularizer."
        )

        self.regularizers = list(regularizers)

    def __len__(self) -> int:
        """
        Length of the 'RegularizerList'.

        Returns
        -------
        length : int
            Length of the list.
        """
        return len(self.regularizers)

    def __iter__(self) -> Iterator:
        """
        Iterates over the regularizers.

        Returns
        -------
        iterator : Iterator
            An iterator over the regularizers.
        """
        return iter(self.regularizers)

    def forward(self, device: torch_device, *args, **kwargs) -> Tensor:
        """
        Computes the forward pass of the regularizer.

        Parameters
        ----------
        device : torch_device
            Torch device.
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The regularization loss.
        """
        if self:
            return sum(stack([regularizer(*args, **kwargs) for regularizer in self.regularizers])).to(device)
        else:
            return tensor(0.0, requires_grad=True).to(device)

    def state_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Collates the states of the regularizers in a dictionary.

        Returns
        -------
        states: Dict[str, Any]
            The state of the regularizers.
        """
        return {regularizer.name: regularizer.state_dict(**kwargs) for regularizer in self.regularizers}


class LpRegularizer(Regularizer):
    """
    Regularizer that applies LP norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            power: int = 1,
            name: Optional[str] = None
    ):
        """
        Constructor of the LP class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        power : int
            The p parameter of the LP norm. Example: p=1 -> L1 norm, p=2 -> L2 norm.
        name : Optional[str]
            The name of the regularizer.
        """
        super().__init__(params=params, lambda_=lambda_, name=name)
        self.power = power

    def forward(self) -> Tensor:
        """
        Computes the forward pass of the regularizer.

        Returns
        -------
        loss : Tensor
            The regularization loss.
        """
        loss = tensor(0.0, requires_grad=True).to(self.params[0].device)
        for param in self.params:
            loss += linalg.norm(param, self.power)
        return loss


class L1Regularizer(LpRegularizer):
    """
    Regularizer that applies L1 norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            name: Optional[str] = None
    ):
        """
        Constructor of the L1 class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        name : Optional[str]
            The name of the regularizer.
        """
        super().__init__(params=params, lambda_=lambda_, name=name, power=1)


class L2Regularizer(LpRegularizer):
    """
    Regularizer that applies L2 norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            name: Optional[str] = None
    ):
        """
        Constructor of the L2 class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        name : Optional[str]
            The name of the regularizer.
        """
        super().__init__(params=params, lambda_=lambda_, name=name, power=2)
