"""
    @file:              table.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 04/2023

    @Description:       This file contains the TableViewer class which is used to visualize a dataset.
"""

from copy import deepcopy
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data.datasets import TableDataset
from ..data.processing.sampling import Mask


class TableViewer:
    """
    A class that is used to visualize tabular data.
    """

    GLOBAL_PATH = "global"
    TASKS_PATH = "task-specific"

    CORRELATIONS_PATH = "correlations"
    FIGURES_PATH = "figures"
    IMPUTED_DF_PATH = "imputed"
    ORIGINAL_DF_PATH = "original"
    TABLES_PATH = "tables"

    def __init__(
            self,
            dataset: TableDataset
    ):
        """
        Sets protected and public attributes of our table viewer.

        Parameters
        ----------
        dataset : TableDataset
            Dataset.
        """
        sns.set_style("whitegrid")
        self.dataset = dataset

        self._target_cols = dataset.target_cols
        self._current_masks = self._get_current_masks()
        self._global_original_df = self._get_original_dataframe()
        self._global_imputed_df = self._get_imputed_dataframe()

        self._nonempty_masks = self._get_nonempty_masks()
        self._task_specific_original_dfs = self._get_original_dataframes()
        self._task_specific_imputed_dfs = self._get_imputed_dataframes()

    def _get_current_masks(self) -> List[Tuple[str, List[int]]]:
        """
        Gets current masks.

        Returns
        -------
        current_masks : List[Tuple[str, List[int]]]
            Current masks.
        """
        masks = [self.dataset.train_mask, self.dataset.valid_mask, self.dataset.test_mask]
        masks_names = [Mask.TRAIN, Mask.VALID, Mask.TEST]
        available_masks = [(name.name, mask) for name, mask in zip(masks_names, masks) if mask]

        return available_masks

    def _get_original_dataframe(self) -> pd.DataFrame:
        """
        Gets original dataframe.

        Returns
        -------
        dataframe : pd.DataFrame
            Original dataframe.
        """
        df_copy = deepcopy(self.dataset.original_data)
        df_copy = pd.concat(
            objs=[
                df_copy.iloc[mask].assign(Sets=name) for name, mask in self._current_masks
            ],
            ignore_index=True
        )
        return df_copy

    def _get_imputed_dataframe(self):
        """
        Gets imputed dataframe.

        Returns
        -------
        dataframe : pd.DataFrame
            Imputed dataframe.
        """
        imputed_df = deepcopy(self.dataset.get_imputed_dataframe())
        imputed_df = pd.concat(
            objs=[
                imputed_df.iloc[mask].assign(Sets=name) for name, mask in self._current_masks
            ],
            ignore_index=True
        )
        return imputed_df

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

        for task_idx, task in enumerate(self.dataset.tasks):
            filtered_masks = deepcopy(self._current_masks)
            for i, (name, mask) in enumerate(filtered_masks):
                nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(self.dataset.y[task.name][mask])
                filtered_masks[i] = (name, np.array(mask)[nonmissing_targets_idx].tolist())

            nonempty_masks.append({name: mask for name, mask in filtered_masks})

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

        for idx, task in enumerate(self.dataset.tasks):
            original_dataframes.append(
                pd.concat(
                    objs=[
                        self.dataset.original_data.iloc[mask].assign(Sets=name) for name, mask in
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

        for idx, task in enumerate(self.dataset.tasks):
            imputed_dataframes.append(
                pd.concat(
                    objs=[
                        self.dataset.get_imputed_dataframe().iloc[mask].assign(Sets=name) for name, mask in
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
        self._create_global_directories(path_to_save)
        self._create_task_specific_directories(path_to_save)

    def _create_global_directories(
            self,
            path_to_save: str
    ) -> None:
        """
        Creates directories used to save tables and figures in the global path.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        global_path = os.path.join(path_to_save, self.GLOBAL_PATH)
        os.makedirs(global_path, exist_ok=True)
        os.makedirs(os.path.join(global_path, self.CORRELATIONS_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.FIGURES_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.FIGURES_PATH, self.ORIGINAL_DF_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.FIGURES_PATH, self.IMPUTED_DF_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.TABLES_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.TABLES_PATH, self.ORIGINAL_DF_PATH), exist_ok=True)
        os.makedirs(os.path.join(global_path, self.TABLES_PATH, self.IMPUTED_DF_PATH), exist_ok=True)

    def _create_task_specific_directories(
            self,
            path_to_save: str
    ) -> None:
        """
        Creates directories used to save tables and figures in the task-specific path.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        os.makedirs(os.path.join(path_to_save, self.TASKS_PATH), exist_ok=True)
        for target_col in self._target_cols:
            os.makedirs(os.path.join(path_to_save, self.TASKS_PATH, target_col), exist_ok=True)
            for path in [self.TABLES_PATH, self.FIGURES_PATH]:
                os.makedirs(os.path.join(path_to_save, self.TASKS_PATH, target_col, path), exist_ok=True)
                for df_path in [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]:
                    os.makedirs(os.path.join(path_to_save, self.TASKS_PATH, target_col, path, df_path), exist_ok=True)

    @staticmethod
    def _save_tables(
            dataframe: pd.DataFrame,
            path_to_save_tables: str
    ) -> None:
        """
        Saves table dataframes.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        path_to_save_tables : str
            Path to save tables.
        """
        dataframe.to_csv(os.path.join(path_to_save_tables, "dataframe.csv"), index=False)
        dataframe.describe().to_csv(os.path.join(path_to_save_tables, "description.csv"), index=False)

    def _save_global_dataframe(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves global dataframe.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        for df, path in zip(
                [self._global_original_df, self._global_imputed_df],
                [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]
        ):
            path_to_save_tables = os.path.join(path_to_save, self.GLOBAL_PATH, self.TABLES_PATH, path)
            self._save_tables(df, path_to_save_tables)

    def _save_task_specific_dataframes(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves task-specific dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        for target, original_df, imputed_df in zip(
                self._target_cols, self._task_specific_original_dfs, self._task_specific_imputed_dfs
        ):
            for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                path_to_save_tables = os.path.join(path_to_save, self.TASKS_PATH, target, self.TABLES_PATH, path)
                self._save_tables(df, path_to_save_tables)

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
        self._save_global_dataframe(path_to_save)
        self._save_task_specific_dataframes(path_to_save)

    def _visualize_correlation(
            self,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = False,
            method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "pearson"
    ) -> None:
        """
        Visualizes correlation.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:
                * pearson : standard correlation coefficient
                * kendall : Kendall Tau correlation coefficient
                * spearman : Spearman rank correlation
                * callable: callable with input two 1d ndarrays
                    and returning a float. Note that the returned matrix from corr
                    will have 1 along the diagonals and will be symmetric
                    regardless of the callable's behavior.
        """
        ds = self.dataset
        sets = [("features", ds.features_cols), ["targets", ds.target_cols], ["all", ds.columns]]
        for name, columns in sets:
            subset = self._global_imputed_df[columns]
            corr = subset.corr(method=method)

            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            corr[mask] = np.nan

            sns.heatmap(
                corr,
                cmap="Greens",
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True
            )

            if path_to_save:
                plt.savefig(os.path.join(path_to_save, self.GLOBAL_PATH, self.CORRELATIONS_PATH, f"{name}.png"))
            if show:
                plt.show()

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

    def _update_axes_of_class_distribution_figure(
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
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            prop={"size": 8}
        )

        plt.setp(autotexts, size=8, weight="bold")

        if title is not None:
            axes.set_title(f"{title} (n = {sum(label_counts.values())})")

    def _visualize_targets(
            self,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = False
    ) -> None:
        """
        Visualizes targets pie charts.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        for i, target_col in enumerate(self._target_cols):
            fig, axes = plt.subplots(ncols=len(self._nonempty_masks[i]), squeeze=False)
            for idx, (name, mask) in enumerate(self._nonempty_masks[i].items()):
                if mask:
                    filtered_df = self._task_specific_original_dfs[i].loc[
                        self._task_specific_original_dfs[i]["Sets"] == name
                        ]
                    self._update_axes_of_class_distribution_figure(
                        axes=axes[0, idx],
                        targets=filtered_df[target_col],
                        label_names={f"{target_col}_0": 0, f"{target_col}_1": 1},
                        title=name
                    )
            fig.tight_layout()
            if path_to_save:
                plt.savefig(os.path.join(path_to_save, self.TASKS_PATH, target_col, self.FIGURES_PATH, "target.png"))
            if show:
                plt.show()
            plt.close(fig)

    @staticmethod
    def _build_cont_features_figure(
            dataframe: pd.DataFrame,
            cont_col: str,
            path_to_save_fig: str,
            show: bool
    ) -> None:
        """
        Builds continuous features figure.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        cont_col : str
            Continuous columns.
        path_to_save_fig : str
            Path to save figures.
        show : bool
            Whether to show figures.
        """
        fig, axes = plt.subplots()
        sns.boxplot(
            data=dataframe,
            y=cont_col,
            x="Sets",
            linewidth=1,
        )
        fig.tight_layout()

        if path_to_save_fig:
            plt.savefig(os.path.join(path_to_save_fig, f"{cont_col}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def _visualize_global_cont_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes global continuous features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        for cont_col in self.dataset.cont_cols:
            for df, path in zip(
                    [self._global_original_df, self._global_imputed_df],
                    [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]
            ):
                path_to_fig = os.path.join(path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, path)
                self._build_cont_features_figure(df, cont_col, path_to_fig, show)

    def _visualize_task_specific_cont_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes task-specific continuous features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        for task, original_df, imputed_df in zip(
                self.dataset.tasks, self._task_specific_original_dfs, self._task_specific_imputed_dfs
        ):
            for cont_col in self.dataset.cont_cols:
                for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                    path_to_fig = os.path.join(
                        path_to_save, self.TASKS_PATH, task.target_column, self.FIGURES_PATH, path
                    )
                    self._build_cont_features_figure(df, cont_col, path_to_fig, show)

    def _visualize_cont_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes continuous features figure.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        self._visualize_global_cont_features(path_to_save, show)
        self._visualize_task_specific_cont_features(path_to_save, show)

    @staticmethod
    def _build_cat_features_figure(
            dataframe: pd.DataFrame,
            cat_col: str,
            path_to_save_fig: str,
            show: bool
    ) -> None:
        """
        Builds categorical features figure.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        cat_col : str
            Categorical column.
        path_to_save_fig : str
            Path to save figures.
        show : bool
            Whether to show figures.
        """
        df_copy = deepcopy(dataframe)
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
        if path_to_save_fig:
            plt.savefig(
                os.path.join(path_to_save_fig, f"{cat_col}.png"), dpi=300)
        if show:
            plt.show()
        plt.close(fig)

    def _visualize_global_cat_features(
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
        for cat_col in self.dataset.cat_cols:
            for df, path in zip(
                    [self._global_original_df, self._global_imputed_df],
                    [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]
            ):
                path_to_fig = os.path.join(path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, path)
                self._build_cat_features_figure(df, cat_col, path_to_fig, show)

    def _visualize_task_specific_cat_features(
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
        for task, original_df, imputed_df in zip(
                self.dataset.tasks, self._task_specific_original_dfs, self._task_specific_imputed_dfs
        ):
            for cat_col in self.dataset.cat_cols:
                for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                    path_to_fig = os.path.join(
                        path_to_save, self.TASKS_PATH, task.target_column, self.FIGURES_PATH, path
                    )
                    self._build_cat_features_figure(df, cat_col, path_to_fig, show)

    def _visualize_cat_features(
            self,
            path_to_save: str,
            show: bool
    ) -> None:
        """
        Visualizes categorical features.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        show : bool
            Whether to show figures.
        """
        self._visualize_global_cat_features(path_to_save, show)
        self._visualize_task_specific_cat_features(path_to_save, show)

    def _visualize_features(
            self,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = False
    ) -> None:
        """
        Visualizes features.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        self._visualize_cont_features(path_to_save, show)
        self._visualize_cat_features(path_to_save, show)

    def visualize(
            self,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = False
    ):
        """
        Visualizes dataset.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        if path_to_save:
            self._create_directories(path_to_save)
            self._save_dataframes(path_to_save)

        self._visualize_targets(path_to_save, show)
        self._visualize_features(path_to_save, show)
        self._visualize_correlation(path_to_save, show)
