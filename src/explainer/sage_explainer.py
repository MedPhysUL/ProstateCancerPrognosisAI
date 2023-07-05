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

import shap
from shap.plots import waterfall, beeswarm, bar, scatter, force
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from monai.data import DataLoader
import numpy as np
import pandas as pd
import sage
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..data.datasets.prostate_cancer import FeaturesType, ProstateCancerDataset, TargetsType
from ..evaluation.prediction_evaluator import PredictionEvaluator
from ..metrics.single_task.base import Direction
from ..models.base.model import Model
from .shap_explainer import CaptumWrapper
from ..tasks.base import TableTask, Task
from ..tasks.containers.list import TaskList
from ..tools.transforms import to_numpy


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
        self.model = model
        self.dataset = dataset
        self.groups = groups

        if groups is None:
            if imputer == "default":
                self.imputer = sage.DefaultImputer(model, values)
            elif imputer == "marginal":
                self.imputer = sage.MarginalImputer(
                    model=model,
                    data=CaptumWrapper.convert_dict_to_tensor(dataset.table_dataset[values].x)
                )
                    # model, wrapper.convert_features_type_to_tuple_of_tensor(FeaturesType(
                    #     (dataset.image_dataset[values].x if dataset.image_dataset is not None else None),
                    #     (dataset.table_dataset[values].x if dataset.table_dataset is not None else None)
                    # )))
        else:
            if imputer == "default":
                self.imputer = sage.GroupedDefaultImputer(model, values, groups=list(groups.values()))
            elif imputer == "marginal":
                self.imputer = sage.GroupedMarginalImputer(model, dataset[values].x, groups=list(groups.values()))

    def plot_sage_values_by_permutation(
            self,
            mask: List[int],
            show: bool,
            path_to_save_folder: Optional[str] = None,
            loss: str = 'cross entropy',
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
        show : bool
            Whether to show the figure.
        path_to_save_folder : Optional[str]
            The path to the folder within which to save the graph, if no path is given then the graph is not saved.
        loss : str
            The loss function to use, either 'mse' or 'cross entropy', defaults to 'cross entropy'.
        n_jobs : int
            Number of parallel processing jobs, defaults to 1.
        random_state : Optional[int]
            The processes' seed, allows for repeatability.
        kwargs
            Kwargs to give to the call function of the estimator, the plot function and to matplotlib.pyplot.savefig.
        """
        estimator = sage.PermutationEstimator(imputer=self.imputer, loss=loss, n_jobs=n_jobs, random_state=random_state)
        sage_values = estimator(
            CaptumWrapper.convert_dict_to_tensor(self.dataset.table_dataset[mask].x),
            CaptumWrapper.convert_dict_to_tensor(self.dataset.table_dataset[mask].y),
            **kwargs
        )
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




