from typing import Optional

import pandas as pd

from .base import Transform


class Normalization(Transform):

    def __call__(self, df: pd.Series, mean: Optional[pd.Series] = None, std: Optional[pd.Series] = None):
        if mean is not None and std is not None:
            return (df-mean)/std
        else:
            return (df-df.mean())/df.std()
