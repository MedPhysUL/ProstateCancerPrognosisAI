"""
    @file:              base.py
    @Author:            Maxence Larose

    @Creation Date:     09/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the abstract `SurvivalAnalysisLoss` class.
"""

from abc import ABC
from typing import Union


from ..base import LossReduction
from ..regression import RegressionLoss
from ....tools.missing_targets import get_idx_of_nonmissing_survival_analysis_targets


class SurvivalAnalysisLoss(RegressionLoss, ABC):
    """
    An abstract class that represents the skeleton of callable classes to use as survival analysis criteria.
    """

    def __init__(
            self,
            name: str,
            reduction: Union[LossReduction, str],
    ):
        """
        Sets protected attributes using parent's constructor

        Parameters
        ----------
        name : str
            Name of the Loss.
        reduction : Union[LossReduction, str]
            Reduction method to use.
        """
        super().__init__(name=name, reduction=reduction)

        self.get_idx_of_nonmissing_targets = get_idx_of_nonmissing_survival_analysis_targets
