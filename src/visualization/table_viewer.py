"""
    @file:              table_viewer.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains the TableViewer class which is used to visualize a dataset.
"""

from copy import deepcopy
import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.processing.multi_task_dataset import MultiTaskDataset
from src.data.processing.single_task_dataset import MaskType, SingleTaskDataset


class TableViewer:

    TABLES_PATH = "tables"
    FIGURES_PATH = "figures"
    ORIGINAL_DF_PATH = "original"
    IMPUTED_DF_PATH = "imputed"

    def __init__(
            self,
            dataset: Union[SingleTaskDataset, MultiTaskDataset]
    ):
        """
        Sets protected and public attributes of our table viewer.

        Parameters
        ----------
        dataset : Union[SingleTaskDataset, MultiTaskDataset]
            Dataset.
        """
        sns.set_style("whitegrid")
        self.dataset = dataset

        if isinstance(dataset, SingleTaskDataset):
            self._datasets = [dataset]
            self._target_cols = [dataset.target_col]
        elif isinstance(dataset, MultiTaskDataset):
            self._datasets = dataset.datasets
            self._target_cols = [ds.target_col for ds in self._datasets]

        self._nonempty_masks = self._get_nonempty_masks()
        self._original_dataframes = self._get_original_dataframes()
        self._imputed_dataframes = self._get_imputed_dataframes()

    def _get_nonempty_masks(
            self
    ) -> List[Dict[str, List[int]]]:
        """
        Gets nonempty masks from all datasets.

        Returns
        -------
        nonempty_masks : List[Dict[str, List[int]]]
            List of nonempty masks dictionaries (one for each of the datasets contained in self._datasets).
        """
        nonempty_masks = []
        for ds in self._datasets:
            available_masks = ds.train_mask, ds.valid_mask, ds.test_mask
            available_masks_names = MaskType.TRAIN, MaskType.VALID, MaskType.TEST
            nonempty_masks.append({name: mask for mask, name in zip(available_masks, available_masks_names) if mask})

        return nonempty_masks

    def _get_original_dataframes(
            self
    ) -> List[Union[pd.Series, pd.DataFrame]]:
        """
        Gets original dataframe from all datasets.

        Returns
        -------
        original_dataframes : List[Union[pd.Series, pd.DataFrame]]
            List of original dataframes (one for each of the datasets contained in self._datasets).
        """
        original_dataframes = []

        for idx, ds in enumerate(self._datasets):
            original_dataframes.append(
                pd.concat(
                    objs=[
                        ds.original_data.iloc[mask].assign(Sets=name) for name, mask in
                        self._nonempty_masks[idx].items()
                    ],
                    ignore_index=True
                )
            )

        return original_dataframes

    def _get_imputed_dataframes(
            self
    ) -> List[Union[pd.Series, pd.DataFrame]]:
        """
        Gets imputed dataframe from all datasets.

        Returns
        -------
        imputed_dataframes : List[Union[pd.Series, pd.DataFrame]]
            List of imputed dataframes (one for each of the datasets contained in self._datasets).
        """
        imputed_dataframes = []

        for idx, ds in enumerate(self._datasets):
            imputed_dataframes.append(
                pd.concat(
                    objs=[
                        ds.get_imputed_dataframe().iloc[mask].assign(Sets=name) for name, mask in
                        self._nonempty_masks[idx].items()
                    ],
                    ignore_index=True
                )
            )

        return imputed_dataframes

    def _create_directories(
            self,
            path_to_save: str
    ) -> None:
        """
        Creates directories used to save tables and figures.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        for target_col in self._target_cols:
            os.makedirs(os.path.join(path_to_save, target_col), exist_ok=True)
            for path in [self.TABLES_PATH, self.FIGURES_PATH]:
                os.makedirs(os.path.join(*[path_to_save, target_col, path]), exist_ok=True)
                for df_path in [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]:
                    os.makedirs(
                        os.path.join(*[path_to_save, target_col, path, df_path]), exist_ok=True
                    )

    def _save_dataframes(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        for target, original_df, imputed_df in zip(
                self._target_cols, self._original_dataframes, self._imputed_dataframes
        ):
            for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                df.to_csv(
                    os.path.join(*[path_to_save, target, self.TABLES_PATH, path, "dataframe"]),
                    index=False
                )
                df.describe().to_csv(
                    os.path.join(*[path_to_save, target, self.TABLES_PATH, path, "description"]),
                    index=False
                )

    @staticmethod
    def _format_to_percentage(
            pct: float,
            values: List[float]
    ) -> str:
        """
        Changes a float to a str representing a percentage.

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
            axes: plt.axes,
            targets: np.array,
            label_names: dict,
            title: Optional[str] = None
    ) -> None:
        """
        Updates axes of a pie chart with classes distribution.

        Parameters
        ----------
        axes : plt.axes
            Axes.
        targets : np.array
            Array of class targets.
        label_names : dict
            Dictionary with names associated to target values.
        title : Optional[str]
            Title for the plot.
        """
        label_counts = {k: np.sum(targets == v) for k, v in label_names.items()}

        labels = [f"{k} ({v})" for k, v in label_counts.items()]

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
            axes.set_title(f"{title} (n = {sum(label_counts.values())})")

    def _visualize_targets(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes targets pie charts.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        for i, target_col in enumerate(self._target_cols):
            fig, axes = plt.subplots(ncols=len(self._datasets), squeeze=False)
            for idx, (name, mask) in enumerate(self._nonempty_masks[i].items()):
                if mask:
                    self._visualize_class_distribution(
                        axes=axes[0, idx],
                        targets=self._original_dataframes[i].iloc[mask][target_col],
                        label_names={f"{target_col}_0": 0, f"{target_col}_1": 1},
                        title=name
                    )
            fig.tight_layout()
            if path_to_save:
                plt.savefig(os.path.join(*[path_to_save, target_col, self.FIGURES_PATH, "target.png"]))
            if show:
                plt.show()
            plt.close(fig)

    def _visualize_cont_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes continuous features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        for ds, original_df, imputed_df in zip(self._datasets, self._original_dataframes, self._imputed_dataframes):
            for cont_col in ds.cont_cols:
                for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                    fig, axes = plt.subplots()
                    sns.boxplot(
                        data=df,
                        y=cont_col,
                        x="Sets",
                        linewidth=1,
                    )
                    fig.tight_layout()

                    if path_to_save:
                        plt.savefig(
                            os.path.join(*[
                                path_to_save, ds.target_col, self.FIGURES_PATH, path, f"{cont_col}.png"
                            ]),
                            dpi=300
                        )
                    if show:
                        plt.show()
                    plt.close(fig)

    def _visualize_cat_features(
            self,
            path_to_save: str,
            show: bool
    ):
        """
        Visualizes categorical features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        for ds, original_df, imputed_df in zip(self._datasets, self._original_dataframes, self._imputed_dataframes):
            for cat_col in ds.cat_cols:
                for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                    df_copy = deepcopy(df)
                    df_copy[cat_col] = df_copy[cat_col].astype("category")

                    fig, axes = plt.subplots()

                    unique = pd.unique(df_copy[cat_col])
                    if all(isinstance(x, (int, float)) for x in unique):
                        axes.set_xticks(pd.unique(df_copy[cat_col]))

                    sns.histplot(
                        data=df_copy,
                        x=cat_col, hue='Sets',
                        multiple='dodge',
                        ax=axes,
                        stat='percent',
                        common_norm=False,
                        shrink=0.8
                    )

                    fig_temp, axes_temp = plt.subplots()
                    sns.histplot(data=df_copy, x=cat_col, hue='Sets', ax=axes_temp)

                    for axes_container, axes_temp_container in zip(axes.containers, axes_temp.containers):
                        labels = [f"{axes_temp_patch.get_height()}" for axes_temp_patch in axes_temp_container]
                        axes.bar_label(axes_container, labels=labels)

                    plt.close(fig_temp)
                    fig.tight_layout()
                    if path_to_save:
                        plt.savefig(
                            os.path.join(*[
                                path_to_save, ds.target_col, self.FIGURES_PATH, path, f"{cat_col}.png"
                            ]),
                            dpi=300
                        )
                    if show:
                        plt.show()
                    plt.close(fig)

    def _visualize_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        self._visualize_cont_features(path_to_save, show)
        self._visualize_cat_features(path_to_save, show)

    def visualize(
            self,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ):
        """
        Visualizes dataset.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        if path_to_save:
            self._create_directories(path_to_save)
            self._save_dataframes(path_to_save)

        self._visualize_targets(path_to_save, show)
        self._visualize_features(path_to_save, show)