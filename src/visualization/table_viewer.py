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
import pandas as pd
import seaborn as sns

from src.data.processing.dataset import MaskType, ProstateCancerDataset


class TableViewer:

    def __init__(
            self,
            dataset: ProstateCancerDataset
    ):
        self.dataset = dataset

    def _get_nonempty_masks(self):
        available_masks = self.dataset.train_mask, self.dataset.valid_mask, self.dataset.test_mask
        available_masks_names = MaskType.TRAIN, MaskType.VALID, MaskType.TEST
        return {name: mask for mask, name in zip(available_masks, available_masks_names) if mask}

    def _get_original_dataframe(self):
        return pd.concat(
            objs=[
                self.dataset.original_data.iloc[mask].assign(Sets=name) for name, mask in
                self._get_nonempty_masks().items()
            ],
            ignore_index=True
        )

    def _get_imputed_dataframe(self):
        return pd.concat(
            objs=[
                self.dataset.get_imputed_dataframe().iloc[mask].assign(Sets=name) for name, mask in
                self._get_nonempty_masks().items()
            ],
            ignore_index=True
        )

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
        for target in self.dataset.target_cols:
            fig, axes = plt.subplots(ncols=len(self._get_nonempty_masks()), squeeze=False)
            for idx, mask in enumerate(self._get_nonempty_masks().values()):
                if mask:
                    self._visualize_class_distribution(
                        fig,
                        axes[0, idx],
                        self.dataset.original_data.iloc[mask][target],
                        label_names={f"{target}_0": 0, f"{target}_1": 1},
                    )
            fig.tight_layout()
            plt.show()

    def _visualize_cont_features(self):
        original_dataframe = self._get_original_dataframe()
        imputed_dataframe = self._get_imputed_dataframe()

        print(original_dataframe.describe())
        print(imputed_dataframe.describe())

        for cont_col in self.dataset.cont_cols:
            fig, axes = plt.subplots()
            sns.boxplot(
                data=original_dataframe,
                y=cont_col,
                x="Sets",
                linewidth=1,
            )
            fig.tight_layout()
            plt.show()

            fig, axes = plt.subplots()
            sns.boxplot(
                data=imputed_dataframe,
                y=cont_col,
                x="Sets",
                linewidth=1,
            )
            fig.tight_layout()
            plt.show()

    def _visualize_cat_features(self):
        original_dataframe = self._get_original_dataframe()
        imputed_dataframe = self._get_imputed_dataframe()

        for cat_col in self.dataset.cat_cols:
            fig, axes = plt.subplots()

            unique = pd.unique(original_dataframe[cat_col])
            if all(isinstance(x, (int, float)) for x in unique):
                axes.set_xticks(pd.unique(original_dataframe[cat_col]))

            sns.histplot(
                data=original_dataframe,
                x=cat_col, hue='Sets',
                multiple='dodge',
                ax=axes,
                stat='probability',
                common_norm=False,
                shrink=0.8
            )
            fig.tight_layout()
            plt.show()

            fig, axes = plt.subplots()

            imputed_dataframe[cat_col] = imputed_dataframe[cat_col].astype("category")
            axes.set_xticks(pd.unique(imputed_dataframe[cat_col]))
            sns.histplot(
                data=imputed_dataframe,
                x=cat_col, hue='Sets',
                multiple='dodge',
                ax=axes,
                stat='probability',
                common_norm=False,
                shrink=0.8
            )
            fig.tight_layout()
            plt.show()

    def _visualize_features(self):
        self._visualize_cont_features()
        self._visualize_cat_features()

    def visualize(self):
        self._visualize_targets()
        self._visualize_features()
