"""
    @file:              regularization.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 12/2022

    @Description:       This file is used to define regularizations which are used to add some penalty to the loss
                        function.
"""

from abc import abstractmethod
from typing import Dict, Iterable, Iterator, Optional, Union

from torch import linalg, tensor, Tensor
from torch.nn import Module, Parameter, ParameterList


class Regularization(Module):
    """
    Base class for regularization.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0
    ):
        """
        Constructor of the BaseRegularization class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super().__init__()

        if isinstance(params, dict):
            params = list(params.values())
        else:
            params = list(params)

        self.params = ParameterList(params)
        self.lambda_ = lambda_

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Call the forward pass of the regularization and scale it by the 'lambda_' attribute.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The loss of the regularization.
        """
        out = super(Regularization, self).__call__(*args, **kwargs)
        return self.lambda_ * out

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """
        Compute the forward pass of the regularization.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The loss of the regularization.
        """
        raise NotImplementedError


class RegularizationList:
    """
    Holds regularizations in a list.
    """

    def __init__(
            self,
            regularizations: Optional[Iterable[Regularization]] = None
    ):
        """
        Constructor of the RegularizationList class.

        Parameters
        ----------
        regularizations : Optional[Iterable[Regularization]]
            The regularizations to apply.
        """
        self.regularizations = regularizations if regularizations is not None else []

    def __iter__(self) -> Iterator:
        """
        Iterate over the regularizations.

        Returns
        -------
        iterator : Iterator
            An iterator over the regularizations.
        """
        return iter(self.regularizations)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Compute the forward pass of the regularization.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The loss of the regularization.
        """
        if len(self.regularizations) == 0:
            return tensor(0)
        loss = sum([regularization(*args, **kwargs) for regularization in self.regularizations])
        return loss


class Lp(Regularization):
    """
    Regularization that applies LP norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            power: int = 1,
            **kwargs
    ):
        """
        Constructor of the L1 class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        power : int
            The p parameter of the LP norm. Example: p=1 -> L1 norm, p=2 -> L2 norm.
        """
        super(Lp, self).__init__(params=params, lambda_=lambda_)
        self.power = power

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Compute the forward pass of the regularization.

        Parameters
        ----------
        args : Any
            Args of the forward pass.
        kwargs : dict
            Kwargs of the forward pass.

        Returns
        -------
        loss : Tensor
            The loss of the regularization.
        """
        loss = tensor(0.0, requires_grad=True).to(self.params[0].device)
        for param in self.params:
            loss += linalg.norm(param, self.power)
        return loss


class L1(Lp):
    """
    Regularization that applies L1 norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0
    ):
        """
        Constructor of the L1 class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super(L1, self).__init__(params=params, lambda_=lambda_, power=1)


class L2(Lp):
    """
    Regularization that applies L2 norm.
    """

    def __init__(
            self,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0
    ):
        """
        Constructor of the L2 class.

        Parameters
        ----------
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super(L2, self).__init__(params=params, lambda_=lambda_, power=2)