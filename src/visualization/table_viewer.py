"""
    @file:              table_viewer.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains the TableViewer class which is used to visualize a prostate cancer dataset.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.processing.dataset import ProstateCancerDataset


class TableViewer:

    def __init__(
            self,
            datasets: List[ProstateCancerDataset]
    ):
        self.datasets = datasets

        if not self._is_datasets_valid():
            raise ValueError("All datasets need to have the same target, continuous data and categorical data columns.")
        else:
            self._target_cols = self.datasets[0].target_cols
            self._cont_cols = self.datasets[0].cont_cols
            self._cat_cols = self.datasets[0].cat_cols

    def _is_datasets_valid(self):
        is_target_cols_valid = all(ds.target_cols == self.datasets[0].target_cols for ds in self.datasets)
        is_cont_cols_valid = all(ds.cont_cols == self.datasets[0].cont_cols for ds in self.datasets)
        is_cat_cols_valid = all(ds.cat_cols == self.datasets[0].cat_cols for ds in self.datasets)

        return all([is_target_cols_valid, is_cont_cols_valid, is_cat_cols_valid])

    @staticmethod
    def _format_to_percentage(
            pct: float,
            values: List[float]
    ) -> str:
        """
        Change a float to a str representing a percentage.

        Parameters
        ----------
        pct : float
            Count related to a class.
        values : List[float]
            Count of items in each class.

        Returns
        -------
        percentage : str
            Percentage as a string.
        """
        absolute = int(round(pct / 100. * np.sum(values)))
        return "{:.1f}%".format(pct, absolute)

    def _visualize_class_distribution(
            self,
            fig,
            axes,
            targets: np.array,
            label_names: dict,
            title: Optional[str] = None
    ) -> None:
        """
        Get fig and axes of a pie chart with classes distribution.

        Parameters
        ----------
        fig
        axes
        targets : np.array
            Array of class targets.
        label_names : dict
            Dictionary with names associated to target values.
        title : Optional[str]
            Title for the plot.
        """
        # We first count the number of instances of each value in the targets vector
        label_counts = {k: np.sum(targets == v) for k, v in label_names.items()}

        # We prepare a list of string to use as plot labels
        labels = [f"{k} ({v})" for k, v in label_counts.items()]

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        wedges, texts, autotexts = axes.pie(
            label_counts.values(),
            textprops=dict(color="w"),
            startangle=90,
            autopct=lambda pct: self._format_to_percentage(pct, list(label_counts.values()))
        )

        axes.legend(
            wedges,
            labels,
            title="Labels",
            loc="center right",
            bbox_to_anchor=(0.1, 0.5, 0, 0),
            prop={"size": 8}
        )

        plt.setp(autotexts, size=8, weight="bold")

        if title is not None:
            axes.set_title(title)

    def _visualize_targets(self):
        for target in self._target_cols:
            fig, axes = plt.subplots()
            for dataset in self.datasets:
                self._visualize_class_distribution(
                    fig,
                    axes,
                    dataset.original_data[target],
                    label_names={"0": 0, "1": 1}
                )
            fig.tight_layout()
            plt.show()

    def _visualize_cont_features(self):
        pass

    def _visualize_cat_features(self):
        pass

    def _visualize_features(self):
        self._visualize_cont_features()
        self._visualize_cat_features()

    def visualize(self):
        self._visualize_targets()
        self._visualize_features()
