"""
    @file:              explainer.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to analyse and explain how a model works.
"""

import json
import os
from typing import Dict, List, Optional, Union, NamedTuple, Tuple

from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..metrics.single_task.base import Direction
from ..models.base.model import Model
from ..tasks.base import TableTask, Task
from ..tasks.containers.list import TaskList
from ..tools.transforms import to_numpy


class CaptumWrapper(torch.nn.Module):

    def __init__(self, model: Model, dataset: ProstateCancerDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.targets_order = [task for task in dataset.tasks]
        self.features_order = {'image': list(dataset[0].x.image.keys()), 'table': list(dataset[0].x.table.keys())}

    def _convert_tensor_to_features_type(self, tensor_tuple: TensorOrTupleOfTensorsGeneric) -> FeaturesType:
        if isinstance(tensor_tuple, tuple):
            tensor_list = list(tensor_tuple)
        else:
            tensor_list = tensor_tuple.tolist()
        image = {}
        table = {}
        for i, key in enumerate(self.features_order['image']):
            image[key] = tensor_list[i]
        for i, key in enumerate(self.features_order['table']):
            i += len(self.features_order['image'])
            table[key] = tensor_list[i]
        return FeaturesType(image=image, table=table)

    def _convert_tensor_to_targets_type(self, tensor_tuple: TensorOrTupleOfTensorsGeneric) -> TargetsType:
        return {task.name: value for task, value in zip(self.targets_order, tensor_tuple)}

    def _convert_features_type_to_tensor(self, features: FeaturesType) -> TensorOrTupleOfTensorsGeneric:
        image_list = [FeaturesType.image[image_key] for image_key in self.features_order['image']]
        table_list = [FeaturesType.table[table_key] for table_key in self.features_order['table']]
        return tuple(image_list + table_list)

    def _convert_targets_type_to_tensor(self, targets: TargetsType) -> TensorOrTupleOfTensorsGeneric:
        targets_list = []
        for task in self.targets_order:
            targets_list.append(targets[task.name])
        return tuple(targets_list)
        # return tuple(targets.values())

    def forward(self, inputs: TensorOrTupleOfTensorsGeneric) -> Tuple[torch.Tensor, ...]:
        inputs = self._convert_tensor_to_features_type(inputs)

        targets_type = self.model(inputs)

        target_tensor = self._convert_targets_type_to_tensor(targets_type)

        return target_tensor


class PredictionModelExplainer:
    """
    This class aims to show how a model works and allow the user to interpret it by using metrics and graphs.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
            mask: List[int]
    ):
        self.model = model
        self.dataset = dataset
        self.mask = mask

    def compute_shap_values(
            self,
            mask
    ):
        raise NotImplementedError
