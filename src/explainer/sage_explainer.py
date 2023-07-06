"""
    @file:              explainer.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to analyse and explain how a model works using sage values.
"""

import json
import os
import re
from typing import Dict, List, Optional, Union, NamedTuple, Tuple

import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
from numpy import concatenate as cat
import sage
import torch

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..models.base.model import Model
from .shap_explainer import CaptumWrapper
from ..tasks.base import TableTask, Task
from ..tasks.survival_analysis import SurvivalAnalysisTask
from ..tasks.containers.list import TaskList
from ..tools.transforms import to_numpy


class SageWrapper(torch.nn.Module):
    """
    A wrapper to allow the usage of sage with the modified models requiring FeaturesType input and having TargetsType
    outputs.
    """
    def __init__(self, model: Model, dataset: ProstateCancerDataset, target: int = None, *args, **kwargs):
        """
        Creates the required variables.

        Parameters
        ----------
        model : Model
            The model to use.
        target: int
            The index of the desired output.
        dataset : ProstateCancerDataset
            The dataset to transform and use as an input.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.target = target
        self.targets_order = [task for task in dataset.tasks]
        self.table_targets_order = [task for task in dataset.tasks.table_tasks]
        self.features_order = {"image": list(dataset[0].x.image.keys()), "table": list(dataset[0].x.table.keys())}

    @staticmethod
    def convert_dict_to_ndarray(
            dictionary: dict,
    ) -> np.ndarray:
        """
        Converts a dictionary of M tensors of size (N, ) into a (N, M) array.

        Parameters
        ----------
        dictionary : dict
            The dictionary to convert.

        Returns
        -------
        tensor : np.ndarray
             The converted array.
        """
        array = []
        for i, key in enumerate(dictionary.keys()):
            datum = dictionary[key]
            if i == 0:
                if isinstance(datum, torch.Tensor):
                    array = to_numpy(datum)
                    if array.ndim == 2:
                        array = array[:, 0]
                elif isinstance(datum, np.ndarray):
                    array = datum
                    if array.ndim == 2:
                        array = array[:, 0]
                else:
                    array = np.array([datum])
                array = np.expand_dims(array, axis=1)
            else:
                if isinstance(datum, torch.Tensor):
                    datum = to_numpy(datum)
                    if datum.ndim == 2:
                        datum = datum[:, 0]
                elif isinstance(datum, np.ndarray):
                    datum = datum
                    if datum.ndim == 2:
                        datum = datum[:, 0]
                else:
                    datum = np.array([datum])
                datum = np.expand_dims(datum, axis=1)
                array = cat((array, datum), axis=1)
        return array

    def _convert_ndarray_to_features_type(
            self,
            table_array: np.ndarray,
            image_array: Optional[np.ndarray] = None
    ) -> FeaturesType:
        """
        Transforms a (1, M) array into a FeaturesType.

        Parameters
        ----------
        table_array : np.ndarray
            The array to convert.

        Returns
        -------
        features : FeaturesType
            The FeaturesType object corresponding to the input data.
        """
        table_array_list = table_array.transpose(*(0, 1)).tolist()  #
        if not isinstance(table_array_list, list):
            table_array_list = [table_array_list]
        if image_array is not None:
            image_array_list = image_array.transpose().tolist()
            if not isinstance(image_array_list, list):
                image_array_list = [image_array_list]
            image = {}
            for i, key in enumerate(self.features_order["image"]):
                image[key] = torch.tensor(image_array_list[i])
        else:
            image = image_array
        table = {}
        for i, key in enumerate(self.features_order["table"]):
            if not isinstance(table_array_list[i][0], int):
                table[key] = torch.tensor(table_array_list[i], dtype=torch.float64)
            else:
                table[key] = torch.tensor(table_array_list[i])
        return FeaturesType(image=image, table=table)

    def _convert_ndarray_to_targets_type(
            self,
            array: np.ndarray
    ) -> TargetsType:
        """
        Transforms a (1, M) array into a TargetsType.

        Parameters
        ----------
        array : np.ndarray
            The array to convert.

        Returns
        -------
        targets : TargetsType
            The TargetsType object corresponding to the input data.
        """
        targets = {}
        for task, value in zip(self.targets_order, array[0].tolist()):
            if not isinstance(value, int):
                targets[task.name] = torch.unsqueeze(torch.tensor(value, dtype=torch.float64), 0)
            else:
                targets[task.name] = torch.unsqueeze(torch.tensor(value), 0)
        return targets

    def _convert_features_type_to_ndarray(
            self,
            features: FeaturesType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms a FeaturesType into a tuple consisting of an (M, ) array comprised of the table data and another
        array comprised of the image data.

        Parameters
        ----------
        features : FeaturesType
            A FeaturesType object to convert into a tuple of arrays.

        Returns
        -------
        array : np.ndarray
            The Tuple of arrays corresponding to the input FeaturesType
        """
        image_list = [features.image[image_key].tolist() for image_key in self.features_order["image"]]
        table_list = []
        for table_key in self.features_order["table"]:
            datum = features.table[table_key]
            if isinstance(datum, (torch.Tensor, np.ndarray)):
                table_list.append(features.table[table_key].tolist())
            else:
                table_list.append([datum])
        return np.array(table_list).transpose(), np.array(image_list).transpose()

    def _convert_targets_type_to_ndarray(
            self,
            targets: TargetsType,
    ) -> np.ndarray:
        """
        Transforms a TargetsType into an array of shape (M, )
        **WARNING: THIS REMOVES THE TIME VALUES ASSOCIATED WITH SURVIVAL EVENTS.**

        Parameters
        ----------
        targets : TargetsType
            A TargetsType object to convert into an array.

        Returns
        -------
        array : np.ndarray
            The array corresponding to the input TargetsType
        """
        targets_list = []
        task_list = self.table_targets_order

        for task in task_list:
            datum = targets[task.name]
            if isinstance(datum, (torch.Tensor, np.ndarray)):
                if datum.ndim == 2:
                    targets_list.append([datum[:, 0]])
                else:
                    targets_list.append(datum.tolist())
            else:
                targets_list.append(np.array([datum]))
        return np.array(targets_list).transpose()

    def forward(self, inputs: np.ndarray) -> torch.Tensor:
        """
        Wrapper around the forward method of the model to use tensors as input and output rather than FeaturesType and
        TargetsType.

        Parameters
        ----------
        inputs : np.ndarray
            Data in the form of tensors or tuple of tensors.

        Returns
        -------
        prediction_tensor : torch.Tensor
            Predictions in the form of a Tensor.
        """
        features_type = self._convert_ndarray_to_features_type(inputs)

        targets_type = self.model(features_type)
        target_array = self._convert_targets_type_to_ndarray(targets_type)
        predictions_array = np.stack(target_array, axis=0)
        if not np.all(np.logical_and(predictions_array >= 0, predictions_array <= 1)):
            prediction_tensor = torch.sigmoid(torch.from_numpy(predictions_array))
        else:
            prediction_tensor = torch.from_numpy(predictions_array)
        return prediction_tensor[:, self.target]


class TableSageValueExplainer:
    """
    This class aims to show how a model works and allow the user to interpret it by using metrics and graphs over a
    whole dataset.
    """

    def __init__(
            self,
            model: Model,
            dataset: ProstateCancerDataset,
            imputer: str,
            values: Union[List[int], FeaturesType],
            groups: Optional[Dict[str, List[List[int]]]] = None
    ):
        """
        Sets required variables

        Parameters
        ----------
        model : Model
            The model to explain.
        dataset : ProstateCancerDataset
            The dataset to use when explaining the model.
        imputer : str
            The type of imputer used, either "default" or "marginal".
        values : Union[List[int], FeaturesType]
            Either the mask to use with the marginal imputer or the default values to use with the default imputer.
        groups : Optional[Dict[str, List[List[int]]]]
            The groups of features for which a singular sage value is desired. If no groups are given then all features
            will have their own sage value computed. This is a dictionary of groups with the keys being the name of the
            groups and the values being the groups. A group being a list of the indices of the features present within
            the group.
        """
        assert dataset.table_dataset is not None, "Sage values require a table dataset to be computed"
        self.wrapper = SageWrapper(model, dataset)
        self.dataset = dataset
        self.model = model
        self.groups = groups

        if groups is None:
            if imputer == "default":
                self.imputer = sage.DefaultImputer(self.wrapper, values)
            elif imputer == "marginal":
                self.imputer = sage.MarginalImputer(
                    model=self.wrapper,
                    data=SageWrapper.convert_dict_to_ndarray(dataset.table_dataset[values].x),
                )
        else:
            if imputer == "default":
                self.imputer = sage.GroupedDefaultImputer(model, values, groups=list(groups.values()))
            elif imputer == "marginal":
                self.imputer = sage.GroupedMarginalImputer(model, dataset[values].x, groups=list(groups.values()))

    def plot_sage_values_by_permutation(
            self,
            mask: List[int],
            target: int,
            show: bool,
            path_to_save_folder: Optional[str] = None,
            loss: str = "cross entropy",
            n_jobs: int = 1,
            random_state: Optional[int] = None,
            **kwargs
    ):
        """
        computes the sage values by permutation of feature indices.

        Parameters
        ----------
        mask : List[int]
            The mask with which to choose the patients used for computation.
        target : int
            The index of the desired output.
        show : bool
            Whether to show the figure.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        loss : str
            The loss function to use, either "mse" or "cross entropy", defaults to "cross entropy".
        n_jobs : int
            Number of parallel processing jobs, defaults to 1.
        random_state : Optional[int]
            The processes' seed, allows for repeatability.
        kwargs
            Kwargs to give to the call function of the estimator, the plot function and to matplotlib.pyplot.savefig.
        """
        self.wrapper.__init__(model=self.model, dataset=self.dataset, target=target)
        estimator = sage.PermutationEstimator(imputer=self.imputer, loss=loss, n_jobs=n_jobs, random_state=random_state)
        sage_values = estimator(
            SageWrapper.convert_dict_to_ndarray(self.dataset.table_dataset[mask].x),
            SageWrapper.convert_dict_to_ndarray(self.dataset.table_dataset[mask].y)[:, target],
            **kwargs
        )
        sage_values.plot(feature_names=self.dataset.table_dataset.features_columns)
        if self.groups is not None:
            fig = sage_values.plot(feature_names=list(self.groups.keys()), return_fig=True,  **kwargs)
        else:
            fig = sage_values.plot(feature_names=self.dataset.table_dataset.features_columns, return_fig=True, **kwargs)

        if path_to_save_folder is not None:
            path = os.path.join(
                path_to_save_folder,
                f"{kwargs.get('filename', 'permutation_sage_values.pdf')}"
            )
        else:
            path = None

        PredictionEvaluator.terminate_figure(fig=fig, show=show, path_to_save_folder=path)
