"""
    @file:              continuous.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 05/2023

    @Description:       This file is used to define the `Normalization` class.
"""

from typing import Optional, Union

import pandas as pd

from .base import Transform


class Normalization(Transform):
    """
    Callable class that performs normalization on continuous data.
    """

    def __call__(
            self,
            df: pd.Series,
            mean: Optional[pd.Series] = None,
            std: Optional[pd.Series] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Performs normalization on continuous data.

        Parameters
        ----------
        df : pd.Series
            Data to transform.
        mean : Optional[pd.Series]
            Mean of the data.
        std : Optional[pd.Series]
            Standard deviation of the data.

        Returns
        -------
        df : Union[pd.DataFrame, pd.Series]
            Transformed data.
        """
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()
