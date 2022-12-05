"""
    @file:              regularization.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 12/2022

    @Description:       This file is used to define regularization callbacks which are used to modify the weights of
                        the model after each iteration according to some penalty.
"""


from abc import abstractmethod
from typing import Dict, Iterable, Iterator, Optional, Union

from torch import linalg, tensor, Tensor
from torch.nn import Module, Parameter, ParameterList
from torch.optim import Optimizer

from src.callbacks.callback import Callback, Priority


class BaseRegularization(Module, Callback):
    """
    Base class for regularization.
    """

    def __init__(
            self,
            name: str,
            optimizer: Optimizer,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            **kwargs
    ):
        """
        Constructor of the BaseRegularization class.

        Parameters
        ----------
        name : str
            The name of the callback.
        optimizer : Optimizer
            The regularization optimizer. It can be different from the actual optimizer used to update the weights
            according to performance.
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super().__init__(**kwargs)
        Callback.__init__(self, name=name, **kwargs)

        if isinstance(params, dict):
            params = list(params.values())
        else:
            params = list(params)

        self.params = ParameterList(params)
        self.lambda_ = lambda_
        self.optimizer = optimizer

    @property
    def priority(self) -> int:
        """
        Priority on a scale from 0 (low priority) to 100 (high priority).

        Returns
        -------
        priority: int
            Callback priority.
        """
        return Priority.MEDIUM_PRIORITY.value

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
        out = super(BaseRegularization, self).__call__(*args, **kwargs)
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

    def on_optimization_end(self, trainer, **kwargs):
        """
        Update weights using the calculated penalty (regularization loss).

        Parameters
        ----------
        trainer : Trainer
            The trainer
        kwargs : dict
            Keywords arguments.
        """
        loss = self()
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()


class RegularizationList(BaseRegularization):
    """
    Holds regularizations in a list.
    """

    def __init__(
            self,
            regularizations: Optional[Iterable[BaseRegularization]] = None,
            **kwargs
    ):
        """
        Constructor of the RegularizationList class.

        Parameters
        ----------
        regularizations : Optional[Iterable[BaseRegularization]]
            The regularizations to apply.
        **kwargs : dict
            The keyword arguments to pass to the Callback.
        """
        self.regularizations = regularizations if regularizations is not None else []
        _params = []
        for regularization in self.regularizations:
            _params.extend(regularization.params)
        super(RegularizationList, self).__init__(
            params=_params,
            lambda_=1.0,
            **kwargs
        )
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


class Lp(BaseRegularization):
    """
    Regularization that applies LP norm.
    """

    def __init__(
            self,
            name: str,
            optimizer: Optimizer,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            power: int = 1,
            **kwargs
    ):
        """
        Constructor of the L1 class.

        Parameters
        ----------
        name : str
            The name of the callback.
        optimizer : Optimizer
            The regularization optimizer. It can be different from the actual optimizer used to update the weights
            according to performance.
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        power : int
            The p parameter of the LP norm. Example: p=1 -> L1 norm, p=2 -> L2 norm.
        """
        super(Lp, self).__init__(name=name, optimizer=optimizer, params=params, lambda_=lambda_, **kwargs)
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
            name: str,
            optimizer: Optimizer,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            **kwargs
    ):
        """
        Constructor of the L1 class.

        Parameters
        ----------
        name : str
            The name of the callback.
        optimizer : Optimizer
            The regularization optimizer. It can be different from the actual optimizer used to update the weights
            according to performance.
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super(L1, self).__init__(name=name, optimizer=optimizer, params=params, lambda_=lambda_, power=1, **kwargs)


class L2(Lp):
    """
    Regularization that applies L2 norm.
    """

    def __init__(
            self,
            name: str,
            optimizer: Optimizer,
            params: Union[Iterable[Parameter], Dict[str, Parameter]],
            lambda_: float = 1.0,
            **kwargs
    ):
        """
        Constructor of the L2 class.

        Parameters
        ----------
        name : str
            The name of the callback.
        optimizer : Optimizer
            The regularization optimizer. It can be different from the actual optimizer used to update the weights
            according to performance.
        params : Union[Iterable[Parameter], Dict[str, Parameter]]
            The parameters which are regularized.
        lambda_ : float
            The weight of the regularization. In other words, the coefficient that multiplies the loss.
        """
        super(L2, self).__init__(name=name, optimizer=optimizer, params=params, lambda_=lambda_, power=2, **kwargs)
