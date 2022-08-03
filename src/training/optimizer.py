"""
    @file:              optimizer.py
    @Author:            Maxence Larose, Nicolas Raymond

    @Creation Date:     05/2022
    @Last modification: 08/2022

    @Description:       This file defines the class associated to Sharpness-Aware Minimization optimizer. The code was
                        mainly taken from : https://github.com/davda54/sam.
"""

from typing import Callable, Iterator, OrderedDict

import torch


class SAM(torch.optim.Optimizer):
    """
    A class that defines a Sharpness-Aware Minimization Optimizer.
    """

    # Constants
    ADAPTIVE = "adaptive"
    PARAMS = "params"
    OLD_P = "old_p"

    def __init__(
            self,
            params: Iterator,
            base_optimizer: Callable,
            rho: float = 0.05,
            adaptive: bool = False,
            **kwargs
    ) -> None:
        """
        Sets private attributes.

        Parameters
        ----------
        params : Iterator
            Model parameters iterator.
        base_optimizer : Callable
            Optimizer constructor function.
        rho : float
            Neighborhood size.
        adaptive : bool
            True to use Adaptive Sharpness-Aware Minimization (ASAM).
        **kwargs : dict
            Other parameters related to base optimizer
        """

        if rho < 0:
            raise ValueError(f"Invalid rho, should be non-negative. rho = {rho}")

        # Call of parent's constructor
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Construction of base optimizer
        self._base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self._base_optimizer.param_groups

    def _grad_norm(
            self
    ) -> torch.Tensor:
        """
        Computes gradients norm.

        Returns
        -------
        norm : torch.Tensor
            Gradient norm.
        """
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0][SAM.PARAMS][0].device

        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group[SAM.ADAPTIVE] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group[SAM.PARAMS]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def first_step(
            self
    ) -> None:
        """
        Performs the first optimization step that finds the weights with the highest loss in the local rho-neighborhood.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group[SAM.PARAMS]:
                if p.grad is not None:
                    self.state[p][SAM.OLD_P] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group[SAM.ADAPTIVE] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"

        self.zero_grad()

    @torch.no_grad()
    def second_step(
            self
    ) -> None:
        """
        Performs the second optimization step that updates the original weights with the gradient from the (locally)
        highest point in the loss landscape.
        """
        for group in self.param_groups:
            for p in group[SAM.PARAMS]:
                if p.grad is not None:
                    p.data = self.state[p][SAM.OLD_P]  # get back to "w" from "w + e(w)"

        self._base_optimizer.step()  # do the actual "sharpness-aware" update
        self.zero_grad()

    def load_state_dict(
            self,
            state_dict: OrderedDict[str, torch.Tensor]
    ) -> None:
        """
        Loads parameters dictionary.

        Parameters
        ----------
        state_dict : OrderedDict[str, torch.Tensor]
            Ordered dictionary with parameters
        """
        super().load_state_dict(state_dict)
        self._base_optimizer.param_groups = self.param_groups
