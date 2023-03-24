"""
    @file:              negative_partial_log_likelihood.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define the Cox `NegativePartialLogLikelihood` class.
"""

from typing import List, Optional, Union

from torch import exp, log, Tensor, unique, where

from ..base import LossReduction
from ..survival_analysis import SurvivalAnalysisLoss


class NegativePartialLogLikelihood(SurvivalAnalysisLoss):
    """
    Callable class that computes the Cox partial negative log-likelihood loss.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            reduction: Union[LossReduction, str] = LossReduction.MEAN
    ):
        """
        Sets protected attributes using parent's constructor.

        Parameters
        ----------
        name : Optional[str]
            Name of the loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction)

        if self.reduction not in (LossReduction.NONE.value, LossReduction.MEAN.value):
            raise ValueError(f"Unsupported reduction: {self.reduction}, available options are ['none', 'mean'].")

    @staticmethod
    def _compute_sorted_partial_negative_log_likelihood(
            pred: Tensor,
            events_indicators: Tensor,
            tied_events_idx: List[List[int]],
            epsilon: float = 1e-7
    ) -> Tensor:
        """
        Computes Cox partial negative log-likelihood. Requires the input to be sorted by descending duration time. The
        Breslow’s method is used for handling tied event times.

        Parameters
        ----------
        pred : Tensor
            (N,) tensor of the natural logarithm of the relative risk function, i.e. g(x) in the original paper.
        events_indicators : Tensor
            (N,) binary tensor of the events' indicators.
        tied_events_idx : List[List[int]]
            List of list of tied events indexes.
        epsilon : float
            Small epsilon for computational stability, i.e. to avoid any division by 0.
        """
        gamma = pred.max()
        cumulative_sum_h = exp(pred - gamma).cumsum(0)
        log_cumulative_sum_h = log(cumulative_sum_h + epsilon) + gamma

        for idx in tied_events_idx:
            log_cumulative_sum_h[idx[:-1]] = log_cumulative_sum_h[idx[-1]].item()

        loss = ((pred - log_cumulative_sum_h) * events_indicators).sum() / events_indicators.sum()

        return - loss

    def _compute_loss(
            self,
            pred: Tensor,
            targets: Tensor
    ) -> Tensor:
        """
        Computes Cox partial negative log-likelihood, where 'pred' are the natural logarithm of the relative risk
        function (g(x) in the original paper), R is the risk set and D is the event occurrence (0 or 1). The risk set R
        can only contain individuals in the current batch. This is a limitation, but simple and fast. The Breslow’s
        method is used for handling tied event times. See equation (8) of (Kvamme, 2019) for a more complete
        mathematical description.

        References:
        [1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel.
            Time-to-event prediction with neural networks and Cox regression.
            Journal of Machine Learning Research, 20(129):1–30, 2019.
            http://jmlr.org/papers/v20/18-424.html
        [2] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger.
            Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.
            BMC Medical Research Methodology, 18(1), 2018.
            https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1

        Parameters
        ----------
        pred : Tensor
            (N,) tensor with predicted labels
        targets : Tensor
            (N, 2) tensor with event indicators and event times.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        events_indicators = targets[:, 0]
        events_times = targets[:, 1]

        events_times, idx = events_times.sort(descending=True)
        events_indicators = events_indicators[idx]
        pred = pred[idx]

        _, inv, counts = unique(events_times, return_inverse=True, return_counts=True)
        tied_events_idx = [where(inv == i)[0].tolist() for i, c, in enumerate(counts) if counts[i] > 1]

        loss = self._compute_sorted_partial_negative_log_likelihood(
            pred=pred,
            events_indicators=events_indicators,
            tied_events_idx=tied_events_idx
        ).to(device=pred.device)

        return loss
