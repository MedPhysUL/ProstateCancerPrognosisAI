"""
    @file:              metrics_evaluator.py
    @Author:            Felix Desroches

    @Creation Date:     06/2023
    @Last modification: 06/2023

    @Description:       This file contains a class used to show metrics and graphs for the human user to gauge the
    quality of a model.
"""

from typing import Dict, List, Optional, Union

from monai.data import DataLoader
from torch import float32, no_grad, random, round, sigmoid, stack, tensor



class Evaluator:
    def __init__(
            self,
            model,
            results,  # Union[Dataset, predictions]
            ground_truth,  # targets?
            mask: List[int] = None
            ):
        """
        Sets the required values for the computation of the different metrics.

        Parameters
        ----------
        Argument : Variables = Default value
            Description

        """

        self.model = model
        self.results = results
        self.ground_truth = ground_truth
        self.mask = mask

    def _dataset_to_predictions(self):

        for features, truth in data_loader:
            prediction = self.model.predict(features=features)


    def show_auc(self, save = False):

    def show_binary_accuracy(self):

    def show_binary_balanced_accuracy(self):

    def show_sensitivity(self):

    def show_specificity(self):

    def show_concordance_index_censored(self):

    def show_breslow(self):



