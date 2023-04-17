"""
    @file:              modality.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This file contains the 'Modality' enum class.
"""

from enum import StrEnum


class Modality(StrEnum):
    """
    Stores the constant related to modality categories.
    """
    CT = "CT"
    PT = "PT"
    MR = "MR"
