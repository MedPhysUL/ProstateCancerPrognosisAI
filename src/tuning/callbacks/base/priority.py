"""
    @file:              priority.py
    @Author:            Maxence Larose

    @Creation Date:     10/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the Priority class used to determine callbacks calling order.
"""

from enum import IntEnum


class Priority(IntEnum):
    LOW_PRIORITY = 0
    MEDIUM_PRIORITY = 50
    HIGH_PRIORITY = 100
