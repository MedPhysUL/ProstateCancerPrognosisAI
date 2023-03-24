"""
    @file:              missing_targets.py
    @Author:            Maxence Larose

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file is used to define different way of getting the indexes of missing targets in a
                        dataset.
"""

from typing import List, Union

import numpy as np
from torch import isnan, Tensor, where


def get_idx_of_nonmissing_classification_targets(
        y: Union[Tensor, np.array]
) -> List[int]:
    """
    Gets the idx of the nonmissing targets in the given array or tensor.

    Parameters
    ----------
    y : Union[Tensor, np.array]
        (N,) tensor or array with targets.

    Returns
    -------
    idx : List[int]
        Index.
    """
    if isinstance(y, Tensor):
        idx = where(y >= 0)
    else:
        idx = np.where(y >= 0)

    return idx[0].tolist()


def get_idx_of_nonmissing_regression_targets(
        y: Union[Tensor, np.array]
) -> List[int]:
    """
    Gets the idx of the nonmissing targets in the given array or tensor.

    Parameters
    ----------
    y : Union[Tensor, np.array]
        (N,) tensor or array with targets.

    Returns
    -------
    idx : List[int]
        Index.
    """
    if isinstance(y, Tensor):
        idx = where(~isnan(y))
    else:
        idx = np.where(~np.isnan(y))

    return idx[0].tolist()


def get_idx_of_nonmissing_survival_analysis_targets(
        y: Union[Tensor, np.array]
) -> List[int]:
    """
    Gets the idx of the nonmissing targets in the given array or tensor.

    Parameters
    ----------
    y : Union[Tensor, np.array]
        (N, 2) tensor or array with targets.

    Returns
    -------
    idx : List[int]
        Index.
    """
    nonmissing_event_indicator_idx = get_idx_of_nonmissing_classification_targets(y[:, 0])
    nonmissing_event_time_idx = get_idx_of_nonmissing_regression_targets(y[:, 1])

    return list(set(nonmissing_event_indicator_idx).intersection(nonmissing_event_time_idx))
