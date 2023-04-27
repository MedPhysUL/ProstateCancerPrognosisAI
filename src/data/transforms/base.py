from abc import ABC, abstractmethod

import pandas as pd


class Transform(ABC):

    @abstractmethod
    def __call__(self, df: pd.Series, **kwargs):
        return NotImplementedError


class Identity(Transform):

    def __call__(self, df: pd.Series, **kwargs):
        return df
