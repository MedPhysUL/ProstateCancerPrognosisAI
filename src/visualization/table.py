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
from sksurv.nonparametric import kaplan_meier_estimator

from ..data.datasets import TableDataset
from ..data.processing.sampling import Mask


class TableViewer:
    """
    A class that is used to visualize tabular data.
    """

    GLOBAL_PATH = "global"
    TARGETS_PATH = "target-specific"

    CORRELATIONS_PATH = "correlations"
    FEATURES_PATH = "features"
    FIGURES_PATH = "figures"
    IMPUTED_DF_PATH = "imputed"
    ORIGINAL_DF_PATH = "original"
    TABLES_PATH = "tables"
    TARGET_PATH = "target"

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

        self._global_masks = self._get_global_masks()
        self._global_original_df = self.get_global_original_dataframe()
        self._global_imputed_df = self.get_global_imputed_dataframe()

        self._target_specific_masks = self._get_target_specific_masks()
        self._target_specific_original_dfs = self._get_target_specific_original_dataframes()
        self._target_specific_imputed_dfs = self._get_target_specific_imputed_dataframes()

    def _get_global_masks(self) -> Dict[str, List[int]]:
        """
        Gets global masks.

        Returns
        -------
        global_masks : List[Tuple[str, List[int]]]
            Global masks of the different sets (train, valid, test, ...) in the dataset.
        """
        masks = [self.dataset.train_mask, self.dataset.valid_mask, self.dataset.test_mask]
        masks_names = [Mask.TRAIN, Mask.VALID, Mask.TEST]
        available_masks = {name: mask for name, mask in zip(masks_names, masks) if mask}

        return available_masks

    def get_global_original_dataframe(self) -> pd.DataFrame:
        """
        Gets global original dataframe.

        Returns
        -------
        dataframe : pd.DataFrame
            Original dataframe.
        """
        df_copy = deepcopy(self.dataset.original_data)
        df_copy = pd.concat(
            objs=[df_copy.iloc[mask].assign(Sets=name) for name, mask in self._global_masks.items()],
            ignore_index=True
        )
        return df_copy

    def get_global_imputed_dataframe(self):
        """
        Gets imputed dataframe.

        Returns
        -------
        dataframe : pd.DataFrame
            Imputed dataframe.
        """
        imputed_df = deepcopy(self.dataset.get_imputed_dataframe())
        imputed_df = pd.concat(
            objs=[imputed_df.iloc[mask].assign(Sets=name) for name, mask in self._global_masks.items()],
            ignore_index=True
        )
        return imputed_df

    def _get_target_specific_masks(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Gets target-specific masks from all the different datasets.

        Returns
        -------
        target_specific_masks : Dict[str, Dict[str, List[int]]]
            Dict of target-specific masks dictionaries.
        """
        nonempty_masks = {}

        for task_idx, task in enumerate(self.dataset.tasks):
            filtered_masks = deepcopy(self._global_masks)
            for name, mask in filtered_masks.items():
                nonmissing_targets_idx = task.get_idx_of_nonmissing_targets(self.dataset.y[task.name][mask])
                filtered_masks[name] = np.array(mask)[nonmissing_targets_idx].tolist()

            nonempty_masks[task.target_column] = {name: mask for name, mask in filtered_masks.items()}

        return nonempty_masks

    def _get_target_specific_original_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Gets target-specific original dataframes from all the different datasets.

        Returns
        -------
        original_target_specific_dataframes : Dict[str, pd.DataFrame]
            Dictionary of target-specific original dataframes.
        """
        original_dataframes = {}

        for task in self.dataset.tasks:
            original_dataframes[task.target_column] = pd.concat(
                objs=[
                    self.dataset.original_data.iloc[mask].assign(Sets=name) for name, mask in
                    self._target_specific_masks[task.target_column].items()
                ],
                ignore_index=True
            )

        return original_dataframes

    def _get_target_specific_imputed_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Gets imputed dataframes.

        Returns
        -------
        imputed_target_specific_dataframes : Dict[str, pd.DataFrame]
            Dictionary of target-specific imputed dataframes.
        """
        imputed_dataframes = {}

        for task in self.dataset.tasks:
            imputed_dataframes[task.target_column] = pd.concat(
                objs=[
                    self.dataset.get_imputed_dataframe().iloc[mask].assign(Sets=name) for name, mask in
                    self._target_specific_masks[task.target_column].items()
                ],
                ignore_index=True
            )

        return imputed_dataframes

    def get_target_specific_original_dataframe(
            self,
            target: str
    ) -> pd.DataFrame:
        """
        Gets target-specific original dataframe.

        Parameters
        ----------
        target : str
            The target column name.

        Returns
        -------
        dataframe : pd.DataFrame
            A dataframe associated to the given target, i.e. a dataframe containing only the patients who have the data
            available for that given target.
        """
        return self._target_specific_original_dfs[target]

    def get_target_specific_imputed_dataframe(
            self,
            target: str
    ):
        """
        Gets target-specific imputed dataframe.

        Parameters
        ----------
        target : str
            The target column name.

        Returns
        -------
        dataframe : pd.DataFrame
            A dataframe associated to the given target, i.e. a dataframe containing only the patients who have the data
            available for that given target.
        """
        return self._target_specific_imputed_dfs[target]

    @staticmethod
    def _terminate_figure(
            path_to_save: Optional[str] = None,
            show: Optional[bool] = None,
            fig: Optional[plt.Figure] = None
    ) -> None:
        """
        Terminates current figure.

        Parameters
        ----------
        path_to_save : Optional[str]
            Path to save.
        show : Optional[bool]
            Whether to show figures.
        fig : Optional[plt.Figure]
            Current figure.
        """
        if fig:
            fig.tight_layout()

        if path_to_save:
            plt.savefig(path_to_save, dpi=300)
        if show:
            plt.show()

        if fig:
            plt.close(fig)
        else:
            plt.close()

    def visualize_correlations(
            self,
            columns: List[str],
            imputed: bool = False,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
            method: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "pearson"
    ) -> None:
        """
        Visualizes correlation.

        Parameters
        ----------
        columns : List[str]
            Features or targets columns to calculate correlations between.
        imputed : bool
            Whether to use imputed or original data.
        path_to_save : Optional[str]
            Path to save figures.
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
        cols = columns + ["Sets"]
        subset = self._global_imputed_df[cols] if imputed else self._global_original_df[cols]

        fig, axes = plt.subplots(ncols=len(self._global_masks), squeeze=False, figsize=(18, 12))

        for idx, (name, mask) in enumerate(self._global_masks.items()):
            filtered_subset = subset.loc[subset["Sets"] == name]
            corr = filtered_subset.corr(method=method, numeric_only=True)

            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            corr[mask] = np.nan

            sns.heatmap(
                corr,
                ax=axes[0, idx],
                cmap="Greens",
                annot=True
            )
            axes[0, idx].set_xticklabels(corr.columns.values, rotation=45, ha='right', rotation_mode='anchor')
            axes[0, idx].set_yticklabels(corr.columns.values, rotation=0)
            axes[0, idx].set_title(f"{name} (n = {len(filtered_subset)})")

        self._terminate_figure(path_to_save, show, fig)

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

        wedges, texts, auto_texts = axes.pie(
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

        plt.setp(auto_texts, size=8, weight="bold")

        if title is not None:
            axes.set_title(f"{title} (n = {sum(label_counts.values())})")

    def visualize_target_class_distribution(
            self,
            target: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ) -> None:
        """
        Visualizes targets pie charts.

        Parameters
        ----------
        target : str
            Target.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        fig, axes = plt.subplots(ncols=len(self._target_specific_masks[target]), squeeze=False)
        for idx, (name, mask) in enumerate(self._target_specific_masks[target].items()):
            if mask:
                filtered_df = self._target_specific_original_dfs[target].loc[
                    self._target_specific_original_dfs[target]["Sets"] == name
                    ]
                self._update_axes_of_class_distribution_figure(
                    axes=axes[0, idx],
                    targets=filtered_df[target],
                    label_names={f"{target}_0": 0, f"{target}_1": 1},
                    title=name
                )
        self._terminate_figure(path_to_save, show, fig)

    def _plot_continuous_feature_figure(
            self,
            dataframe: pd.DataFrame,
            feature: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = None
    ) -> None:
        """
        Plots a continuous feature figure.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        feature : str
            The name of the column containing the continuous feature data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        fig, axes = plt.subplots()
        sns.boxplot(
            data=dataframe,
            y=feature,
            x="Sets",
            linewidth=1,
        )
        self._terminate_figure(path_to_save, show, fig)

    def visualize_global_continuous_feature(
            self,
            feature: str,
            imputed: bool = False,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ) -> None:
        """
        Visualizes a continuous feature figure from the global dataframe.

        Parameters
        ----------
        feature : str
            The name of the column containing the continuous feature data.
        imputed : bool
            Whether to use imputed or original data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        dataset = self._global_imputed_df if imputed else self._global_original_df
        self._plot_continuous_feature_figure(dataset, feature, path_to_save, show)

    def visualize_target_specific_continuous_features(
            self,
            feature: str,
            target: str,
            imputed: bool = False,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ) -> None:
        """
        Visualizes a continuous feature figure from a target-specific dataframe.

        Parameters
        ----------
        feature : str
            The name of the column containing the continuous feature data.
        target : str
            The name of the column containing the target data.
        imputed : bool
            Whether to use imputed or original data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        dataset = self._target_specific_imputed_dfs[target] if imputed else self._target_specific_original_dfs[target]
        self._plot_continuous_feature_figure(dataset, feature, path_to_save, show)

    def _plot_categorical_feature_figure(
            self,
            dataset: pd.DataFrame,
            feature: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Plots a categorical feature figure.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe.
        feature : str
            The name of the column containing the categorical feature data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        df_copy = deepcopy(dataset)
        df_copy[feature] = df_copy[feature].astype("category")

        fig, axes = plt.subplots()

        unique = pd.unique(df_copy[feature])
        if all(isinstance(x, (int, float)) for x in unique):
            axes.set_xticks(pd.unique(df_copy[feature]))

        sns.histplot(
            data=df_copy,
            x=feature,
            hue='Sets',
            multiple='dodge',
            ax=axes,
            stat='percent',
            common_norm=False,
            shrink=0.8
        )

        fig_temp, axes_temp = plt.subplots()
        sns.histplot(data=df_copy, x=feature, hue='Sets', ax=axes_temp)

        for axes_container, axes_temp_container in zip(axes.containers, axes_temp.containers):
            labels = [f"{axes_temp_patch.get_height()}" for axes_temp_patch in axes_temp_container]
            axes.bar_label(axes_container, labels=labels)

        plt.close(fig_temp)
        self._terminate_figure(path_to_save, show, fig)

    def visualize_global_categorical_feature(
            self,
            feature: str,
            imputed: bool = False,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Visualizes a categorical feature figure from the global dataframe.

        Parameters
        ----------
        feature : str
            The name of the column containing the categorical feature data.
        imputed : bool
            Whether to use imputed or original data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        dataset = self._global_imputed_df if imputed else self._global_original_df
        self._plot_categorical_feature_figure(dataset, feature, path_to_save, show)

    def visualize_target_specific_categorical_features(
            self,
            feature: str,
            target: str,
            imputed: bool = False,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Visualizes a categorical feature figure from a target-specific dataframe.

        Parameters
        ----------
        feature : str
            The name of the column containing the categorical feature data.
        target : str
            The name of the column containing the target data.
        imputed : bool
            Whether to use imputed or original data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        dataset = self._target_specific_imputed_dfs[target] if imputed else self._target_specific_original_dfs[target]
        self._plot_categorical_feature_figure(dataset, feature, path_to_save, show)

    def _plot_global_kaplan_meier_curve(
            self,
            dataset: pd.DataFrame,
            event_indicator: str,
            event_time: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Plots the global kaplan meier curve (without stratification).

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe.
        event_indicator : str
            The name of the column containing the event indicator data.
        event_time : str
            The name of the column containing the event time data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        fig, axes = plt.subplots()

        for idx, (name, mask) in enumerate(self._global_masks.items()):
            filtered_dataset = dataset.loc[dataset["Sets"] == name]
            time, survival_probability = kaplan_meier_estimator(
                filtered_dataset[event_indicator].astype(bool),
                filtered_dataset[event_time]
            )
            axes.step(time, survival_probability, where="post", label=name)

        plt.ylabel(f"Estimated probability of {event_indicator} survival")
        plt.xlabel("Time")
        plt.legend()

        self._terminate_figure(path_to_save, show, fig)

    def _plot_stratified_kaplan_meier_curve(
            self,
            dataset: pd.DataFrame,
            event_indicator: str,
            event_time: str,
            feature: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Plots the global kaplan meier curve (without stratification).

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe.
        event_indicator : str
            The name of the column containing the event indicator data.
        event_time : str
            The name of the column containing the event time data.
        feature : str
            The name of the column containing the feature data used for stratification.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.

        Returns
        -------
        figure : plt.Figure
            A figure.
        """
        df_copy = deepcopy(dataset)
        df_copy[feature] = df_copy[feature].astype("category")

        unique = pd.unique(df_copy[feature])
        values_count = df_copy[feature].value_counts()
        unique_categories = [category for category in unique.categories if values_count[category] >= 1]

        fig, axes = plt.subplots()
        line_styles = ["solid", "dashed", "dotted"]

        cm = plt.get_cmap('gist_rainbow')
        axes.set_prop_cycle(color=[cm(1. * i / len(unique_categories)) for i in range(len(unique_categories))])
        for idx, (name, mask) in enumerate(self._global_masks.items()):
            filtered_df = df_copy.loc[dataset["Sets"] == name]
            unique = pd.unique(filtered_df[feature])
            values_count = filtered_df[feature].value_counts()
            unique_categories = [category for category in unique.categories if values_count[category] >= 1]
            for category in unique_categories:
                mask = filtered_df[feature] == category
                time, survival_probability = kaplan_meier_estimator(
                    filtered_df[event_indicator][mask].astype(bool),
                    filtered_df[event_time][mask]
                )
                axes.step(
                    time, survival_probability, where="post", label=f"{name};{category}", linestyle=line_styles[idx]
                )

        plt.ylabel(f"Estimated probability of {event_indicator} survival")
        plt.xlabel("Time")
        plt.legend(loc="best")

        self._terminate_figure(path_to_save, show, fig)

    def visualize_kaplan_meier_curve(
            self,
            event_indicator: str,
            event_time: str,
            imputed: bool = False,
            feature: Optional[str] = None,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True,
    ) -> None:
        """
        Visualizes a categorical feature figure from a target-specific dataframe.

        Parameters
        ----------
        event_indicator : str
            The name of the column containing the event indicator data.
        event_time : str
            The name of the column containing the event time data.
        imputed : bool
            Whether to use imputed or original data.
        feature : Optional[str]
            The name of the column containing the categorical feature data used for stratification.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        if imputed:
            dataset = self._target_specific_imputed_dfs[event_indicator]
        else:
            dataset = self._target_specific_original_dfs[event_indicator]

        if feature:
            self._plot_stratified_kaplan_meier_curve(dataset, event_indicator, event_time, feature, path_to_save, show)
        else:
            self._plot_global_kaplan_meier_curve(dataset, event_indicator, event_time, path_to_save, show)

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

        figures_path = os.path.join(global_path, self.FIGURES_PATH)
        os.makedirs(figures_path, exist_ok=True)

        tables_path = os.path.join(global_path, self.TABLES_PATH)
        correlations_path = os.path.join(figures_path, self.CORRELATIONS_PATH)
        features_path = os.path.join(figures_path, self.FEATURES_PATH)
        for path in [correlations_path, features_path, tables_path]:
            os.makedirs(path, exist_ok=True)
            os.makedirs(os.path.join(path, self.ORIGINAL_DF_PATH), exist_ok=True)
            os.makedirs(os.path.join(path, self.IMPUTED_DF_PATH), exist_ok=True)

    def _create_target_specific_directories(
            self,
            path_to_save: str
    ) -> None:
        """
        Creates directories used to save tables and figures in the target-specific path.

        Parameters
        ----------
        path_to_save : str
            Path to save descriptive analysis records.
        """
        os.makedirs(os.path.join(path_to_save, self.TARGETS_PATH), exist_ok=True)
        for task in self.dataset.tasks:
            target_path = os.path.join(path_to_save, self.TARGETS_PATH, task.target_column)
            os.makedirs(target_path, exist_ok=True)

            figures_path = os.path.join(target_path, self.FIGURES_PATH)
            os.makedirs(figures_path, exist_ok=True)

            tables_path = os.path.join(target_path, self.TABLES_PATH)
            features_path = os.path.join(figures_path, self.FEATURES_PATH)
            for path in [tables_path, features_path]:
                os.makedirs(path, exist_ok=True)
                os.makedirs(os.path.join(path, self.ORIGINAL_DF_PATH), exist_ok=True)
                os.makedirs(os.path.join(path, self.IMPUTED_DF_PATH), exist_ok=True)

            sub_target_path = os.path.join(figures_path, self.TARGET_PATH)
            os.makedirs(sub_target_path, exist_ok=True)
            if task in self.dataset.tasks.survival_analysis_tasks:
                os.makedirs(os.path.join(sub_target_path, self.ORIGINAL_DF_PATH), exist_ok=True)
                os.makedirs(os.path.join(sub_target_path, self.IMPUTED_DF_PATH), exist_ok=True)

    def _create_directories(
            self,
            path_to_save: str
    ) -> None:
        """
        Creates directories used to save tables and figures.

        Parameters
        ----------
        path_to_save : str
            Path to save the descriptive analysis records.
        """
        self._create_global_directories(path_to_save)
        self._create_target_specific_directories(path_to_save)

    @staticmethod
    def _save_tables(
            dataframe: pd.DataFrame,
            path_to_save: str
    ) -> None:
        """
        Saves dataframes in the .csv format.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        path_to_save : str
            Path to save tables.
        """
        dataframe.to_csv(os.path.join(path_to_save, "dataframe.csv"), index=False)
        dataframe.describe().to_csv(os.path.join(path_to_save, "description.csv"), index=False)

    def _save_global_dataframe(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves global original and imputed dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save global dataframes.
        """
        for df, path in zip(
                [self._global_original_df, self._global_imputed_df],
                [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]
        ):
            path_to_save_tables = os.path.join(path_to_save, self.GLOBAL_PATH, self.TABLES_PATH, path)
            self._save_tables(df, path_to_save_tables)

    def _save_target_specific_dataframes(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves target-specific original and imputed dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save target-specific dataframes.
        """
        for (target, original_df), (target, imputed_df) in zip(
                self._target_specific_original_dfs.items(), self._target_specific_imputed_dfs.items()
        ):
            for df, path in zip([original_df, imputed_df], [self.ORIGINAL_DF_PATH, self.IMPUTED_DF_PATH]):
                path_to_save_tables = os.path.join(path_to_save, self.TARGETS_PATH, target, self.TABLES_PATH, path)
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
            Path to save all dataframes.
        """
        self._save_global_dataframe(path_to_save)
        self._save_target_specific_dataframes(path_to_save)

    def _save_correlation_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves correlations figures.

        Parameters
        ----------
        path_to_save : str
            Path to save correlations figures.
        """
        ds = self.dataset
        sets = [("features", ds.features_cols), ["targets", ds.target_cols], ["all", ds.columns]]
        for imputed, path in [(False, self.ORIGINAL_DF_PATH), (True, self.IMPUTED_DF_PATH)]:
            for name, columns in sets:
                path_to_fig = os.path.join(
                    path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, self.CORRELATIONS_PATH, path, f"{name}.png"
                )
                self.visualize_correlations(columns, imputed, path_to_fig, False)

    def _save_targets_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves targets figures.

        Parameters
        ----------
        path_to_save : str
            Path to save targets figures.
        """
        for i, target in enumerate(self._target_cols):
            path = os.path.join(
                path_to_save, self.TARGETS_PATH, target, self.FIGURES_PATH, self.TARGET_PATH, "class_distribution.png"
            )
            self.visualize_target_class_distribution(target, path, False)

    def _save_global_features_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves features from the global original and imputed dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save global original and imputed dataframes.
        """
        for imputed, path in [(False, self.ORIGINAL_DF_PATH), (True, self.IMPUTED_DF_PATH)]:
            directory = os.path.join(path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, self.FEATURES_PATH, path)
            for cont_col in self.dataset.cont_features_cols:
                path_to_fig = os.path.join(directory, f"{cont_col}.png")
                self.visualize_global_continuous_feature(cont_col, imputed, path_to_fig, False)
            for cat_col in self.dataset.cat_features_cols:
                path_to_fig = os.path.join(directory, f"{cat_col}.png")
                self.visualize_global_categorical_feature(cat_col, imputed, path_to_fig, False)

    def _save_target_specific_features_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves features from the target-specific original and imputed dataframes.

        Parameters
        ----------
        path_to_save : str
            Path to save target-specific original and imputed dataframes.
        """
        for target in self._target_cols:
            for imputed, path in [(False, self.ORIGINAL_DF_PATH), (True, self.IMPUTED_DF_PATH)]:
                directory = os.path.join(
                    path_to_save, self.TARGETS_PATH, target, self.FIGURES_PATH, self.FEATURES_PATH, path
                )
                for cont_col in self.dataset.cont_features_cols:
                    path_to_fig = os.path.join(directory, f"{cont_col}.png")
                    self.visualize_target_specific_continuous_features(cont_col, target, imputed, path_to_fig, False)
                for cat_col in self.dataset.cat_features_cols:
                    path_to_fig = os.path.join(directory, f"{cat_col}.png")
                    self.visualize_target_specific_categorical_features(cat_col, target, imputed, path_to_fig, False)

    def _save_features_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves features figures.

        Parameters
        ----------
        path_to_save : str
            Path to save featires figures.
        """
        self._save_global_features_figures(path_to_save)
        self._save_target_specific_features_figures(path_to_save)

    def _save_kaplan_meier_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves kaplan meier figures for survival analysis task.

        Parameters
        ----------
        path_to_save : str
            Path to save kaplan meier figures.
        """
        for task in self.dataset.tasks.survival_analysis_tasks:
            for imputed, path in [(False, self.ORIGINAL_DF_PATH), (True, self.IMPUTED_DF_PATH)]:
                directory = os.path.join(
                    path_to_save, self.TARGETS_PATH, task.target_column, self.FIGURES_PATH, self.TARGET_PATH, path
                )
                event, time = task.event_indicator_column, task.event_time_column
                path_to_fig = os.path.join(directory, "survival (global).png")
                self.visualize_kaplan_meier_curve(event, time, imputed, None, path_to_fig, False)

                for cat_col in self.dataset.cat_features_cols:
                    path_to_fig = os.path.join(directory, f"survival({cat_col}).png")
                    self.visualize_kaplan_meier_curve(event, time, imputed, cat_col, path_to_fig, False)

    def save_descriptive_analysis(
            self,
            path_to_save: str
    ):
        """
        Saves descriptive analysis.

        Parameters
        ----------
        path_to_save : str
            Path to save the descriptive analysis records.
        """
        self._create_directories(path_to_save)

        self._save_dataframes(path_to_save)
        self._save_correlation_figures(path_to_save)
        self._save_targets_figures(path_to_save)
        self._save_features_figures(path_to_save)
        self._save_kaplan_meier_figures(path_to_save)
