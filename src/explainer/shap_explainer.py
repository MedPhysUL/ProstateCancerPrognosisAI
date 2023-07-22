"""
    @file:              shap_explainer.py
    @Author:            Félix Desroches

    @Creation Date:     06/2023
    @Last modification: 07/2023

    @Description:       This file contains a class used to analyse and explain how a model works using shapley values.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

from captum.attr import IntegratedGradients
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
import shap
import torch

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..models.base.model import Model
from ..tools.plot import terminate_figure


class CaptumWrapper(torch.nn.Module):
    """
    A wrapper to allow the usage of captum with the modified models requiring FeaturesType input and having TargetsType
    outputs.
    """

    def __init__(self, model: Model, dataset: ProstateCancerDataset):
        """
        Creates the required variables.

        Parameters
        ----------
        model : Model
            The model to use.
        dataset : ProstateCancerDataset
            The dataset to transform and use as an input.
        """
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.features_order = {"image": list(dataset[0].x.image.keys()), "table": list(dataset[0].x.table.keys())}

    def _convert_tensor_to_features_type(
            self,
            tensor_tuple: TensorOrTupleOfTensorsGeneric
    ) -> FeaturesType:
        """
        Transforms a (1, M) tensor or a tuple of M tensors of shape (1, ) into a FeaturesType.

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
            while not isinstance(tensor_list[0], torch.Tensor):
                temp_list = tensor_list
                tensor_list = []
                for item in temp_list:
                    tensor_list += [datum for datum in item]
        else:
            tensor_list = tensor_tuple.tolist()
        image, table = {}, {}
        for i, key in enumerate(self.features_order["image"]):
            image[key] = tensor_list[i]
        for i, key in enumerate(self.features_order["table"]):
            i += len(self.features_order["image"])
            table[key] = tensor_list[i]
        return FeaturesType(image=image, table=table)

    def _convert_tensor_to_targets_type(
            self,
            tensor_tuple: TensorOrTupleOfTensorsGeneric
    ) -> TargetsType:
        """
        Transforms a (1, M) tensor or a tuple of M tensors of shape (1, ) into a TargetsType.

        Parameters
        ----------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The tensor or the tuple of tensors to convert.

        Returns
        -------
        targets : TargetsType
            The TargetsType object corresponding to the input data.
        """
        return {task.name: value for task, value in zip(self.dataset.tasks.tasks, tensor_tuple)}

    def _convert_features_type_to_tuple_of_tensor(
            self,
            features: FeaturesType
    ) -> Tuple[torch.Tensor, ...]:
        """
        Transforms a FeaturesType into a tuple of M tensors of shape (1, ).

        Parameters
        ----------
        features : FeaturesType
            A FeaturesType object to convert into a tuple of tensors.

        Returns
        -------
        tensor_tuple : TensorOrTupleOfTensorsGeneric
            The Tuple of tensors corresponding to the input FeaturesType
        """
        image_list = [features.image[image_key] for image_key in self.features_order["image"]]
        table_list = []
        for table_key in self.features_order["table"]:
            datum = features.table[table_key]
            if isinstance(datum, torch.Tensor):
                table_list.append(features.table[table_key])
            elif isinstance(datum, np.ndarray):
                table_list.append((torch.from_numpy(datum)))
            else:
                table_list.append(torch.tensor([datum]))
        return tuple(image_list + table_list)

    def _convert_targets_type_to_tuple_of_tensor(
            self,
            targets: TargetsType,
            ignore_seg_tasks: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Transforms a TargetsType into a tuple of M tensors of shape (1, ).

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
        task_list = self.dataset.tasks.table_tasks.tasks if ignore_seg_tasks else self.dataset.tasks.tasks

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

        Parameters
        ----------
        inputs : TensorOrTupleOfTensorsGeneric
            Data in the form of tensors or tuple of tensors.

        Returns
        -------
        prediction_tensor : torch.Tensor
            Predictions in the form of a Tensor.
        """
        if not isinstance(inputs, tuple):
            inputs = self._convert_tensor_to_features_type(tuple([*inputs]))
        else:
            inputs = self._convert_tensor_to_features_type(inputs)

        predictions = {}
        outputs = self.model(inputs)
        for task in self.dataset.tasks.binary_classification_tasks:
            predictions[task.name] = torch.sigmoid(outputs[task.name])
        for task in self.dataset.tasks.regression_tasks:
            predictions[task.name] = outputs[task.name]
        for task in self.dataset.tasks.survival_analysis_tasks:
            predictions[task.name] = outputs[task.name]
        for task in self.dataset.tasks.segmentation_tasks:
            predictions[task.name] = torch.round(torch.sigmoid(outputs[task.name]))
        target_tensor = self._convert_targets_type_to_tuple_of_tensor(predictions, ignore_seg_tasks=True)
        prediction_tensor = torch.stack(target_tensor, dim=1)

        return prediction_tensor


class TableShapValueExplainer:
    """
    This class aims to show how a model works and allow the user to interpret it by using metrics and graphs.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
    ) -> None:
        """
        Sets the required variables of the class.

        Parameters
        ----------
        model : Model
            The model to explain.
        dataset : ProstateCancerDataset
            The dataset with which to explain the model.
        """
        assert dataset.image_dataset is None and dataset.table_dataset is not None, (
            "SHAP values require a table dataset and cannot be computed with a model that requires an image dataset"
        )
        self.model = CaptumWrapper(model, dataset)
        self.old_model = model
        self.dataset = dataset
        self.base_values = self.compute_shap_base_values()

    def compute_shap_values(
            self,
            target: int,
            mask: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Computes the shap values for the tabular data used within the model.

        Parameters
        ----------
        target : int
            Index of the output for which the shap values are desired.
        mask : Optional[List[int]]
            Mask to select the patients used when computing the shap values.

        Returns
        -------
        shap_values : np.ndarray
            Array of the shap values for the patients, with the first dimension being the patients and the second the
            features.
        """
        integrated_gradient = IntegratedGradients(self.model)
        rng_state = torch.random.get_rng_state()
        subset = self.dataset if mask is None else self.dataset[mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        torch.random.set_rng_state(rng_state)
        cated_tensor_list_to_cat = []
        for features, _ in data_loader:
            features = tuple([feature.requires_grad_() for feature in features.table.values()])
            attr = integrated_gradient.attribute(features, target=target)
            tensor_list_to_cat = []
            for tensor in attr:
                tensor_list_to_cat.append(tensor)
            cated_tensor = torch.cat(tensor_list_to_cat)
            cated_tensor_list_to_cat.append(torch.unsqueeze(cated_tensor, 0))
        attr_tensor = torch.cat(cated_tensor_list_to_cat)
        return attr_tensor.detach().numpy()

    def compute_average_shap_values(
            self,
            targets: Union[int, List[int]],
            mask: Optional[List[int]] = None,
            absolute: bool = False,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> Dict[str, float]:
        """
        Computes the average shap values and shows them in a graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        absolute : bool
            Whether to compute the absolute value before computing the average. Defaults to False.
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.

        Returns
        -------
        average_shap : Dict[str, float]
            A dictionary of the average shap value for each feature.
        """
        if not isinstance(targets, list):
            targets = [targets]
        average_attributions = {}
        feature_names = []

        for target in targets:
            fig, arr = plt.subplots()
            feature_names = self.dataset.table_dataset.features_columns
            if absolute:
                average_attributions = {
                    target: np.mean(np.abs(self.compute_shap_values(target=target, mask=mask)), axis=0)
                    for target in targets
                }
            else:
                average_attributions = {
                    target: np.mean(self.compute_shap_values(target=target, mask=mask), axis=0)
                    for target in targets
                }

            title = kwargs.get("title", "Average Feature Importances")
            axis_title = kwargs.get("axis", "Features")
            x_pos = (np.arange(len(feature_names)))

            arr.bar(x_pos, average_attributions[target], align="center")
            arr.set_xticks(x_pos)
            arr.set_xticklabels(feature_names, minor=False, rotation=90)
            arr.set_xlabel(axis_title)
            arr.set_title(title)

            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                file_name = "average_absolute_shap_plot.pdf" if absolute else "average_shap_plot.pdf"
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', file_name)}"
                )
            else:
                path = None

            terminate_figure(fig=fig, show=show, path_to_save=path, **kwargs)

        average_shap = {
            target: {feature_names[i]: average_attributions[target][i] for i in range(len(feature_names))}
            for target in targets
        }
        return average_shap

    def compute_shap_base_values(self):
        """
        Computes the base values for shap as the mean of the model's outputs with the training dataset.

        Returns
        -------
        base_values : torch.Tensor
            The base values.
        """
        rng_state = torch.random.get_rng_state()
        subset = self.dataset[self.dataset.train_mask]
        data_loader = DataLoader(dataset=subset, batch_size=1, shuffle=False, collate_fn=None)
        torch.random.set_rng_state(rng_state)
        tensor_list_to_cat = []
        for features, targets in data_loader:
            features = tuple([feature.requires_grad_() for feature in features.table.values()])
            pred = self.model.forward(features)
            tensor_list_to_cat.append(pred)
        return torch.mean(torch.cat(tensor_list_to_cat, dim=0), dim=0)

    def plot_force(
            self,
            targets: Union[int, List[int]],
            patient_id: int,
            mask: Optional[List[int]] = None,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the force graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        patient_id : int
            The index of the patient for whom to compute the graph.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        show : bool
            Whether to show the graph, defaults to True.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target, mask=mask)
            data = self.dataset.table_dataset.x[mask] if mask is not None else self.dataset.table_dataset.x
            shap_values = shap.Explanation(
                values=values,
                base_values=float(self.base_values[target]),
                feature_names=self.dataset.table_dataset.features_columns,
                data=data
            )

            shap.force_plot(shap_values[patient_id], matplotlib=True, show=False)
            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'force_plot.pdf')}"
                )
            else:
                path = None
            terminate_figure(show=show, path_to_save=path, **kwargs)

    def plot_waterfall(
            self,
            targets: Union[int, List[int]],
            patient_id: int,
            mask: Optional[List[int]] = None,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        patient_id : int
            The index of the patient for whom to compute the graph.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        show : bool
            Whether to show the graph, defaults to True.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target, mask=mask)
            shap_values = shap.Explanation(
                values=values,
                feature_names=self.dataset.table_dataset.features_columns,
                base_values=float(self.base_values[target]),
                data=self.dataset.table_dataset.x
            )
            shap.plots.waterfall(shap_values[patient_id], show=False)
            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'waterfall_plot.pdf')}"
                )
            else:
                path = None
            terminate_figure(show=show, path_to_save=path, **kwargs)

    def plot_beeswarm(
            self,
            targets: Union[int, List[int]],
            mask: Optional[List[int]] = None,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        show : bool
            Whether to show the graph, defaults to True.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target, mask=mask)
            data = self.dataset.table_dataset.x[mask] if mask is not None else self.dataset.table_dataset.x
            shap_values = shap.Explanation(
                values=values,
                base_values=float(self.base_values[target]),
                feature_names=self.dataset.table_dataset.features_columns,
                data=data
            )
            shap.plots.beeswarm(shap_values, show=False)
            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'beeswarm_plot.pdf')}"
                )
            else:
                path = None
            terminate_figure(show=show, path_to_save=path, **kwargs)

    def plot_bar(
            self,
            targets: Union[int, List[int]],
            mask: Optional[List[int]] = None,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        show : bool
            Whether to show the graph, defaults to True.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target, mask=mask)
            data = self.dataset.table_dataset.x[mask] if mask is not None else self.dataset.table_dataset.x
            shap_values = shap.Explanation(
                values=values,
                base_values=np.ones_like(values)*float(self.base_values[target]),
                feature_names=self.dataset.table_dataset.features_columns,
                data=data
            )
            shap.plots.bar(shap_values, show=False)
            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'bar_plot.pdf')}"
                )
            else:
                path = None
            terminate_figure(show=show, path_to_save=path, **kwargs)

    def plot_scatter(
            self,
            targets: Union[int, List[int]],
            mask: Optional[List[int]] = None,
            show: bool = True,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        mask : Optional[List[int]]
            A mask to select which patients to use.
        show : bool
            Whether to show the graph, defaults to True.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target, mask=mask)
            data = self.dataset.table_dataset.x[mask] if mask is not None else self.dataset.table_dataset.x
            shap_values = shap.Explanation(
                values=values,
                base_values=np.ones_like(values)*float(self.base_values[target]),
                feature_names=self.dataset.table_dataset.features_columns,
                data=data
            )
            shap.plots.scatter(shap_values, show=False)
            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'scatter_plot.pdf')}"
                )
            else:
                path = None
            terminate_figure(show=show, path_to_save=path, **kwargs)
