"""
    @file:              continuous.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 05/2023

    @Description:       This file is used to define the `Normalization` class.
"""

from typing import Optional

import pandas as pd

from .base import Transform


class Normalization(Transform):

    def __call__(self, df: pd.Series, mean: Optional[pd.Series] = None, std: Optional[pd.Series] = None):
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()
