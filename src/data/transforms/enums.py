from enum import Enum

from .base import Identity
from .continuous import Normalization
from .categorical import OneHotEncoding, OrdinalEncoding, MappingEncoding


class ContinuousTransform(Enum):
    """
    Enum class that defines all continuous transformations.
    """
    IDENTITY = Identity
    NORMALIZATION = Normalization


class CategoricalTransform(Enum):
    """
    Enum class that defines all categorical transformations.
    """
    IDENTITY = Identity
    ONE_HOT_ENCODING = OneHotEncoding
    ORDINAL_ENCODING = OrdinalEncoding
    MAPPING_ENCODING = MappingEncoding
