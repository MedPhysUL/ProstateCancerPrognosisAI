from typing import Hashable, Mapping, Union

import pandas as pd

from .base import Transform


class OneHotEncoding(Transform):

    def __call__(self, df: pd.Series, **kwargs):
        return pd.get_dummies(df)


class OrdinalEncoding(Transform):

    def __call__(self, df: pd.Series, **kwargs):
        return df.cat.codes


class MappingEncoding(Transform):

    def __init__(self, mapping: Mapping[Hashable, Union[float, int]]):
        self.mapping = mapping

    def __call__(self, df: pd.Series, **kwargs):
        return df.map(self.mapping)
