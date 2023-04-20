"""
    @file:              negative_partial_log_likelihood.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define the Cox `NegativePartialLogLikelihood` class.
"""

from typing import Optional, Union

from torch import exp, log, ones, sum, tensor, Tensor, unsqueeze

from ..base import LossReduction
from ..survival_analysis import SurvivalAnalysisLoss


class NegativePartialLogLikelihood(SurvivalAnalysisLoss):
    """
    Callable class that computes the Cox partial negative log-likelihood loss.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            reduction: Union[LossReduction, str] = LossReduction.NONE
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

    def _compute_loss(
            self,
            pred: Tensor,
            event_indicator: Tensor,
            event_time: Tensor
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
            (N,) tensor with predicted labels.
        event_indicator : Tensor
            (N,) tensor with event indicators.
        event_time : Tensor
            (N,) tensor with event times.

        Returns
        -------
        loss : Tensor
            Loss value.
        """
        if event_indicator.count_nonzero() == 0:
            return tensor(0.0, device=pred.device)

        event_time_length = event_time.shape[0]
        mask = ones(event_time_length, event_time_length).to(device=pred.device)
        event_time_matrix = unsqueeze(event_time, 0)
        mask[(event_time_matrix.T - event_time_matrix) > 0] = 0

        hazard_ratio = exp(pred) * mask
        log_risk = log(sum(hazard_ratio, dim=1))
        uncensored_likelihood = pred - log_risk
        censored_likelihood = uncensored_likelihood * event_indicator
        neg_log_loss = -sum(censored_likelihood) / sum(event_indicator)

        return neg_log_loss.to(device=pred.device)
