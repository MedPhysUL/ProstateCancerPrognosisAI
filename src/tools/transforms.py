"""
    @file:              transforms.py
    @Author:            Maxence Larose

    @Creation Date:     12/2022
    @Last modification: 02/2023

    @Description:       This file is used to define the different transforms.
"""

import numbers
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from torch import device as torch_device
from torch import float32, from_numpy, tensor, Tensor
from torch.nn import Module

from ..data.datasets.prostate_cancer import FeaturesType


def batch_to_device(
        batch: Union[dict, FeaturesType, Tensor],
        device: torch_device
) -> Union[dict, FeaturesType, Tensor]:
    """
    Sends batch to device.

    Parameters
    ----------
    batch : Union[dict, FeaturesType, Tensor]
        Batch data.
    device : torch_device
        Torch device
    """
    if isinstance(batch, FeaturesType):
        image_features = {k: batch_to_device(v, device) for k, v in batch.image.items()}
        table_features = {k: batch_to_device(v, device) for k, v in batch.table.items()}
        return FeaturesType(image=image_features, table=table_features)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, Tensor):
        return batch.to(device)
    return batch


def to_tensor(x: Any, dtype=float32):
    if isinstance(x, np.ndarray):
        return from_numpy(x).type(dtype)
    elif isinstance(x, Tensor):
        return x.type(dtype)
    elif isinstance(x, pd.DataFrame):
        return to_tensor(to_numpy(x, dtype=np.float32), dtype)
    elif not isinstance(x, Tensor):
        return tensor(x, dtype=dtype)
    raise ValueError(f"Unsupported type {type(x)}.")


def to_numpy(x: Any, dtype=np.float32):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype)
    elif isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype)
    elif isinstance(x, numbers.Number):
        return x
    elif isinstance(x, dict):
        return {k: to_numpy(v, dtype=dtype) for k, v in x.items()}
    elif not isinstance(x, Tensor):
        return np.asarray(x, dtype=dtype)
    raise ValueError(f"Unsupported type {type(x)}.")


class ToDevice(Module):
    def __init__(self, device: torch_device, non_blocking: bool = True):
        super().__init__()
        self.device = device
        self.non_blocking = non_blocking

    def forward(self, x: Tensor):
        if x is None:
            return x
        if not isinstance(x, Tensor):
            if isinstance(x, dict):
                return {k: self.forward(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [self.forward(v) for v in x]
            elif isinstance(x, tuple):
                return tuple(self.forward(v) for v in x)
            else:
                return x
        return x.to(self.device, non_blocking=self.non_blocking)


class ToTensor(Module):
    def __init__(self, dtype=float32, device: Optional[torch_device] = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.to_device = ToDevice(self.device) if device else None

    def forward(self, x: Any) -> Tensor:
        x = to_tensor(x, self.dtype)
        x = self.to_device(x) if self.to_device else x
        return x
