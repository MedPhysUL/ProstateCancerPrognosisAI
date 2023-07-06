"""
    @file:              shap_explainer.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to analyse and explain how a model works using shapley values.
"""

import os
from typing import Dict, List, Optional, Union, Tuple

import shap
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
import torch

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..models.base.model import Model


class CaptumWrapper(torch.nn.Module):
    """
    A wrapper to allow the usage of captum with the modified models requiring FeaturesType input and having TargetsType
    outputs.
    """

    def __init__(self, model: Model, dataset: ProstateCancerDataset, *args, **kwargs):
        """
        Creates the required variables.

        Parameters
        ----------
        model : Model
            The model to use.
        dataset : ProstateCancerDataset
            The dataset to transform and use as an input.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.targets_order = [task for task in dataset.tasks]
        self.table_targets_order = [task for task in dataset.tasks.table_tasks]
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
        else:
            tensor_list = tensor_tuple.tolist()
        image = {}
        table = {}
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
        return {task.name: value for task, value in zip(self.targets_order, tensor_tuple)}

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

        Parameters
        ----------
        inputs : TensorOrTupleOfTensorsGeneric
            Data in the form of tensors or tuple of tensors.

        Returns
        -------
        prediction_tensor : torch.Tensor
            Predictions in the form of a Tensor.
        """
        inputs = self._convert_tensor_to_features_type(tuple([*inputs]))
        predictions ={}
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
        assert dataset.table_dataset is not None, "Shap values require a table dataset to be computed"
        self.model = CaptumWrapper(model, dataset)
        self.dataset = dataset

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

        n = 0
        attr_tensor = torch.tensor([])
        for features, _ in data_loader:
            features = tuple([feature.requires_grad_() for feature in features.table.values()])
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

            if path_to_save_folder is not None:
                path_to_save_folder = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('filename', 'average_shap_values.pdf')}"
                )
            title = kwargs.get("title", "Average Feature Importances")
            axis_title = kwargs.get("axis", "Features")
            x_pos = (np.arange(len(feature_names)))

            arr.bar(x_pos, average_attributions[target], align="center")
            arr.set_xticks(x_pos, feature_names, wrap=True)
            arr.set_xlabel(axis_title)
            arr.set_title(title)

            if path_to_save_folder is not None:
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'average_shap_plot.pdf')}"
                )
            else:
                path = None

            PredictionEvaluator.terminate_figure(fig=fig, show=show, path_to_save_folder=path)

        average_shap = {
            target: {feature_names[i]: average_attributions[target][i] for i in range(len(feature_names))}
            for target in targets
        }
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
            targets: Union[int, List[int]],
            patient_id: int,
            show: bool,
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
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target)
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
                target_names = self.dataset.table_dataset.target_columns
                path = os.path.join(
                    path_to_save_folder,
                    f"{kwargs.get('target', target_names[target])}_{kwargs.get('filename', 'force_plot.pdf')}"
                )
            else:
                path = None
            self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_waterfall(
            self,
            targets: Union[int, List[int]],
            patient_id: int,
            show: bool,
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
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target)
            shap_values = shap.Explanation(
                values=values,
                base_values=0
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
            self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_beeswarm(
            self,
            targets: Union[int, List[int]],
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target)
            shap_values = shap.Explanation(
                values=values,
                base_values=0,
                feature_names=self.dataset.table_dataset.features_columns,
                data=self.dataset.table_dataset.x
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
            self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_bar(
            self,
            targets: Union[int, List[int]],
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target)
            shap_values = shap.Explanation(
                values=values,
                base_values=np.zeros_like(values),
                feature_names=self.dataset.table_dataset.features_columns,
                data=self.dataset.table_dataset.x
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
            self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)

    def plot_scatter(
            self,
            targets: Union[int, List[int]],
            show: bool,
            path_to_save_folder: Optional[str] = None,
            **kwargs
    ) -> None:
        """
        Computes the waterfall graph.

        Parameters
        ----------
        targets : Union[int, List[int]]
            The index or a list of the indexes of the desired output for which to compute the shap values.
        show : bool
            Whether to show the graph
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        kwargs
            Kwargs to give to matplotlib.pyplot.savefig.
        """
        if isinstance(targets, int):
            targets = [targets]
        for target in targets:
            values = self.compute_shap_values(target=target)
            shap_values = shap.Explanation(
                values=values,
                base_values=np.zeros_like(values),
                feature_names=self.dataset.table_dataset.features_columns,
                data=self.dataset.table_dataset.x
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
            self.terminate_figure(show=show, path_to_save_folder=path, **kwargs)
