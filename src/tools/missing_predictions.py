"""
    @file:              missing_predictions.py
    @Author:            Maxence Larose

    @Creation Date:     05/2022
    @Last modification: 05/2023

    @Description:       This file is used to define different way of getting the indexes of missing predictions in the
                        output of a model.
"""

from typing import List, Union

import numpy as np
from torch import isnan, Tensor, where


def get_idx_of_nonmissing_predictions(
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
