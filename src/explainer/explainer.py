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

import shap
from shap.plots import waterfall, beeswarm, bar, scatter, force
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..metrics.single_task.base import Direction
from ..models.base.model import Model
from ..tasks.base import TableTask, Task
from ..tasks.containers.list import TaskList
from ..tools.transforms import to_numpy


class CaptumWrapper(torch.nn.Module):
    """

    """

    def __init__(self, model: Model, dataset: ProstateCancerDataset, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.targets_order = [task for task in dataset.tasks]
        self.table_targets_order = [task for task in dataset.tasks.table_tasks]
        self.features_order = {'image': list(dataset[0].x.image.keys()), 'table': list(dataset[0].x.table.keys())}

    def convert_tensor_to_features_type(
            self,
            tensor_tuple: TensorOrTupleOfTensorsGeneric
    ) -> FeaturesType:
        """
        Transforms a (N, M) tensor or a tuple of M (N, ) tensors into a FeaturesType.

        Parameters
        ----------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The tensor or the tuple of tensors to convert.

        Returns
        -------
        features : FeaturesType
            The FeaturesType object corresponding to the input data.
        """
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

    def convert_tensor_to_targets_type(
            self,
            tensor_tuple: TensorOrTupleOfTensorsGeneric
    ) -> TargetsType:
        """
        Transforms a (N, M) tensor or a tuple of M (N, ) tensors into a TargetsType.

        Parameters
        ----------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The tensor or the tuple of tensors to convert.

        Returns
        -------
        targets : TargetsType
            The TargetsType object corresponding to the input data.
        """
        return {task.name: value for task, value in zip(self.targets_order, tensor_tuple)}

    def convert_features_type_to_tuple_of_tensor(
            self,
            features: FeaturesType
    ) -> Tuple[torch.Tensor, ...]:
        """
        Transforms a FeaturesType into a tuple of M (N, ) tensors.

        Parameters
        ----------
        features : FeaturesType
            A FeaturesType object to convert into a tuple of tensors.

        Returns
        -------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The Tuple of tensors corresponding to the input FeaturesType
        """
        image_list = [features.image[image_key] for image_key in self.features_order['image']]
        table_list = []
        for table_key in self.features_order['table']:
            datum = features.table[table_key]
            if isinstance(datum, torch.Tensor):
                table_list.append(features.table[table_key])
            elif isinstance(datum, np.ndarray):
                table_list.append((torch.from_numpy(datum)))
            else:
                table_list.append(torch.tensor([datum]))
        # table_list = [features.table[table_key] for table_key in self.features_order['table']]
        return tuple(image_list + table_list)

    def convert_targets_type_to_tuple_of_tensor(
            self,
            targets: TargetsType,
            ignore_seg_tasks: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Transforms a TargetsType into a tuple of M (N, ) tensors.

        Parameters
        ----------
        targets : TargetsType
            A TargetsType object to convert into a tuple of tensors.
        ignore_seg_tasks : bool
            Whether to ignore the seg tasks when converting to a tensor.

        Returns
        -------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The tuple of tensors corresponding to the input TargetsType
        """
        targets_list = []
        if ignore_seg_tasks:
            task_list = self.table_targets_order
        else:
            task_list = self.targets_order

        for task in task_list:
            datum = targets[task.name]
            if isinstance(datum, torch.Tensor):
                targets_list.append(datum)
            elif isinstance(datum, np.ndarray):
                targets_list.append(torch.from_numpy(datum))
            else:
                targets_list.append(torch.tensor([datum]))
        return tuple(targets_list)

    def forward(self, *inputs: TensorOrTupleOfTensorsGeneric) -> torch.Tensor:
        """
        Wrapper around the forward method of the model to use tensors as input and output rather than FeaturesType and
        TargetsType.
        """
        inputs = self.convert_tensor_to_features_type(tuple([*inputs]))

        targets_type = self.model(inputs)
        target_tensor = self.convert_targets_type_to_tuple_of_tensor(targets_type, ignore_seg_tasks=True)
        target_tensor = torch.stack(target_tensor, dim=0)

        return target_tensor


class PredictionModelExplainer:
    """
    This class aims to show how a model works and allow the user to interpret it by using metrics and graphs.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
    ):
        self.model = CaptumWrapper(model, dataset)
        self.dataset = dataset

    def compute_table_shap_values(
            self,
            target,
            mask: Optional[List[int]] = None
    ):
        """

        """
        integrated_gradient = IntegratedGradients(self.model)
        rng_state = torch.random.get_rng_state()
        subset = self.dataset if mask is None else self.dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        torch.random.set_rng_state(rng_state)

        n = 0
        attr_tensor = torch.tensor([])
        for features, targets in data_loader:
            features = self.model.convert_features_type_to_tuple_of_tensor(features)
            features = tuple([feature.requires_grad_() for feature in features])
            attr = integrated_gradient.attribute(features, target=target)
            cat_tensor = torch.tensor([])

            for i, tensor in enumerate(attr):
                if i == 0:
                    cat_tensor = tensor
                else:
                    cat_tensor = torch.cat((cat_tensor, tensor))
            if n == 0:
                attr_tensor = torch.unsqueeze(cat_tensor, 0)
            else:
                attr_tensor = torch.cat((attr_tensor, torch.unsqueeze(cat_tensor, 0)))
            n += 1
        return attr_tensor.detach().numpy()

    def compute_average_shap_values(
            self,
            target: int, # si list de int, avoir un subplot de tous les targets
            mask: Optional[List[int]],
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        """

        """
        fig, arr = plt.subplots()
        feature_names = self.dataset.table_dataset.features_columns
        average_attributions = np.mean(self.compute_table_shap_values(target=target, mask=mask), axis=0)

        if path_to_save_folder is not None:
            path_to_save_folder = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('filename', 'average_shap_values.pdf')}"
            )
        title = kwargs.get('title', "Average Feature Importances")
        axis_title = kwargs.get('axis', "Features")
        x_pos = (np.arange(len(feature_names)))

        arr.bar(x_pos, average_attributions, align='center')
        arr.set_xticks(x_pos, feature_names, wrap=True)
        arr.set_xlabel(axis_title)
        arr.set_title(title)

        PredictionEvaluator.terminate_figure(fig=fig, show=show, path_to_save_folder=path_to_save_folder)

        average_shap = {feature_names[i]: average_attributions[i] for i in range(len(feature_names))}
        return average_shap

    @staticmethod
    def terminate_figure(
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Terminates current figure.

        Parameters
        ----------
        path_to_save_folder : Optional[str]
            Path to save the figure.
        show : bool
            Whether to show figure.
        """

        if path_to_save_folder is not None:
            plt.savefig(path_to_save_folder, **kwargs)
        if show:
            plt.show()
        plt.close()

    def plot_force(
            self,
            target: int,
            patient_id: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        values = self.compute_table_shap_values(target=target)
        shap_values = shap.Explanation(
            values=values,
            base_values=np.zeros_like(values),
            feature_names=self.dataset.table_dataset.features_columns,
            data=self.dataset.table_dataset.x
        )

        shap.force_plot(shap_values[patient_id],
                        matplotlib=True,
                        show=False
                        )
        if path_to_save_folder is not None:
            features = self.dataset.table_dataset.features_columns
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('target', features[target])}_{kwargs.get('filename', 'force_plot.pdf')}"
            )
        else:
            path = None
        self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_waterfall(
            self,
            target: int,
            patient_id: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        values = self.compute_table_shap_values(target=target)
        shap_values = shap.Explanation(
            values=values,
            base_values=0
        )
        shap.plots.waterfall(shap_values[patient_id], show=False)
        if path_to_save_folder is not None:
            features = self.dataset.table_dataset.features_columns
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('target', features[target])}_{kwargs.get('filename', 'waterfall_plot.pdf')}"
            )
        else:
            path = None
        self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_beeswarm(
            self,
            target: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        values = self.compute_table_shap_values(target=target)
        shap_values = shap.Explanation(
            values=values,
            base_values=0,
            feature_names=self.dataset.table_dataset.features_columns,
            data=self.dataset.table_dataset.x
        )
        shap.plots.beeswarm(shap_values, show=False)
        if path_to_save_folder is not None:
            features = self.dataset.table_dataset.features_columns
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('target', features[target])}_{kwargs.get('filename', 'beeswarm_plot.pdf')}"
            )
        else:
            path = None
        self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_bar(
            self,
            target: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        values = self.compute_table_shap_values(target=target)
        shap_values = shap.Explanation(
            values=values,
            base_values=np.zeros_like(values),
            feature_names=self.dataset.table_dataset.features_columns,
            data=self.dataset.table_dataset.x
        )
        shap.plots.bar(shap_values, show=False)
        if path_to_save_folder is not None:
            features = self.dataset.table_dataset.features_columns
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('target', features[target])}_{kwargs.get('filename', 'bar_plot.pdf')}"
            )
        else:
            path = None
        self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_scatter(
            self,
            target: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ):
        values = self.compute_table_shap_values(target=target)
        shap_values = shap.Explanation(
            values=values,
            base_values=np.zeros_like(values),
            feature_names=self.dataset.table_dataset.features_columns,
            data=self.dataset.table_dataset.x
        )
        shap.plots.scatter(shap_values, show=False)
        if path_to_save_folder is not None:
            features = self.dataset.table_dataset.features_columns
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('target', features[target])}_{kwargs.get('filename', 'scatter_plot.pdf')}"
            )
        else:
            path = None
        self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)
