"""
    @file:              table.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 04/2023

    @Description:       This file contains the TableViewer class which is used to visualize a dataset.
"""

from copy import deepcopy
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgError
from scipy.interpolate import interp1d
from scipy.stats import chi2_contingency, mannwhitneyu
import seaborn as sns
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

from .color import Color, DarkColor, LightColor
from ..data.datasets import TableDataset
from ..data.processing.sampling import Mask
from .tools import add_at_risk_counts, survival_table_from_events
from ..tools.plot import add_details_to_kaplan_meier_curve, terminate_figure


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
            dataset: TableDataset,
            feature_names: Optional[Dict[str, str]] = None,
            target_names: Optional[Dict[str, str]] = None,
            crop: Optional[Dict[str, Tuple[Tuple[Optional[float], float], Tuple[Optional[float], float]]]] = None,
            fig_size: Tuple[int, int] = (8, 6)
    ):
        """
        Sets protected and public attributes of our table viewer.

        Parameters
        ----------
        dataset : TableDataset
            Dataset.
        feature_names : Dict[str, str]
            Dictionary of feature names. The keys are the original feature names and the values are the feature names
            to be displayed.
        target_names : Dict[str, str]
            Dictionary of target names. The keys are the original target names and the values are the target names to
            be displayed.
        crop : Optional[Dict[str, Tuple[Tuple[Optional[float], float], Tuple[Optional[float], float]]]]
            Dictionary of crop values. The keys are the original feature names and the values are the crop values. The
            first value is the one associated to the original dataframe and the second value is the one associated to
            the imputed dataframe.
        fig_size : Tuple[int, int]
            Figure size.
        """
        # sns.set_style("white")
        matplotlib.rc('axes', edgecolor='k')
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'

        self.dataset = dataset
        self._feature_names = feature_names
        self._target_names = target_names
        self._crop = crop
        self._target_cols = dataset.target_columns

        self._global_masks = self._get_global_masks()
        self._global_original_df = self.get_global_original_dataframe()
        self._global_imputed_df = self.get_global_imputed_dataframe()

        self._target_specific_masks = self._get_target_specific_masks()
        self._target_specific_original_dfs = self._get_target_specific_original_dataframes()
        self._target_specific_imputed_dfs = self._get_target_specific_imputed_dataframes()

        full_legend_names = {
            Mask.TRAIN.value: "Training set",
            Mask.VALID.value: "Validation set",
            Mask.TEST.value: "Test set"
        }
        partial_legend_names = {
            Mask.TRAIN.value: "Learning set",
            Mask.TEST.value: "Holdout set"
        }
        self._legend_names = partial_legend_names if len(self._global_masks) == 2 else full_legend_names
        self._fig_size = fig_size

        self._colors = [c for c in Color]
        self._light_colors = [c for c in LightColor]
        self._dark_colors = [c for c in DarkColor]
        sns.set_palette([self._light_colors[i] for i in range(len(self._global_masks))])

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
        df_copy = deepcopy(self.dataset.dataframe)
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
        imputed_df = deepcopy(self.dataset.imputed_dataframe)
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
                    self.dataset.dataframe.iloc[mask].assign(Sets=name) for name, mask in
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
                    self.dataset.imputed_dataframe.iloc[mask].assign(Sets=name) for name, mask in
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

    def _get_sets_dict(
            self,
            dataframe: pd.DataFrame
    ):
        """
        Gets sets dictionary.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.

        Returns
        -------
        sets_dict : Dict[str, pd.DataFrame]
            Sets dictionary.
        """
        sets = {
            "full_dataset": dataframe,
            Mask.TRAIN: dataframe[dataframe["Sets"] == Mask.TRAIN],
            Mask.VALID: dataframe[dataframe["Sets"] == Mask.VALID],
            Mask.TEST: dataframe[dataframe["Sets"] == Mask.TEST]
        }
        return {k: v for k, v in sets.items() if not v.empty}

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

        dataframes = self._get_sets_dict(subset)
        for idx, (name, df) in enumerate(dataframes.items()):
            fig, axes = plt.subplots(figsize=self._fig_size)
            colors = [(1, 1, 1), self._light_colors[idx]]
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Custom", colors, 256)

            corr = df.corr(method=method, numeric_only=True)

            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask)] = True
            corr[mask] = np.nan

            heatmap = sns.heatmap(
                corr,
                cmap=cmap,
                annot=True,
                annot_kws={"size": 14, "color": "k"},
            )
            colorbar = heatmap.collections[0].colorbar
            colorbar.ax.tick_params(labelsize=14)
            colorbar.ax.set_ylabel("Pearson correlation", fontsize=16)

            col_values = [self._target_names[n] if n in self._target_names else n for n in corr.columns.values]
            axes.set_xticklabels(col_values, rotation=45, ha="right", rotation_mode="anchor", fontsize=13)
            axes.set_yticklabels(col_values, rotation=0, fontsize=13)

            terminate_figure(path_to_save=os.path.join(path_to_save, f"{name}.png"), show=show, fig=fig)

    def visualize_timeline(
            self,
            name: str,
            dataframe: pd.DataFrame,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ) -> None:
        """
        Visualizes timeline.

        Parameters
        ----------
        name : str
            Name of the figure.
        dataframe : pd.DataFrame
            Dataframe.
        path_to_save : Optional[str]
            Path to save figures.
        show : Optional[bool]
            Whether to show figures.
        """
        plt.figure(figsize=self._fig_size)

        # order = ["BCR", "METASTASIS", "HTX", "CRPC", "DEATH"]
        # sorted_tasks = sorted(self.dataset.tasks.survival_analysis_tasks, key=lambda o: order.index(o.target_column))
        sorted_tasks = self.dataset.tasks.survival_analysis_tasks
        dataframes, tasks = [], []
        for idx, task in enumerate(sorted_tasks):
            event_column, time_column = task.event_indicator_column, task.event_time_column
            df = dataframe[dataframe[event_column] == 1][[time_column]]
            df.rename(columns={time_column: "time"}, inplace=True)
            dataframes.append(df)
            tasks.append(task.event_indicator_column)
        dataframe = pd.concat(dataframes, keys=tasks)
        dataframe.reset_index(inplace=True)

        g = sns.FacetGrid(dataframe, row="level_0", hue="level_0", aspect=12, height=5.0, palette=self._light_colors)
        g.map(sns.swarmplot, "time", color="k", size=12, alpha=1.0)
        g.map(sns.kdeplot, "time", clip_on=False, fill=True, linewidth=1.5, alpha=1, cut=0, bw_adjust=0.25)

        def label(x, color, label):
            axes = plt.gca()
            axes.set_facecolor("none")
            axes.invert_yaxis()
            label = self._target_names[label] if label in self._target_names.keys() else label
            axes.text(
                -0.04, .40, label, fontweight="bold", color="k", ha="left", va="center", transform=axes.transAxes,
                fontsize=66
            )
            axes.axhline(y=0.0, xmin=-0.04, xmax=1.0, color="k", linestyle="-", linewidth=3, clip_on=False)
            # if label == "PCSS":
            #     axes.tick_params(axis="x", color="k", which="major", labelsize=64, length=16)
            #     axes.spines["bottom"].set_position(("axes", 0.25))

        g.map(label, "time")
        g.set_titles("")
        g.figure.subplots_adjust(hspace=-0.25)
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        g.set_xlabels("Time $($months$)$", fontsize=74)
        terminate_figure(path_to_save=os.path.join(path_to_save, f"{name}.png"), show=show)

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
        return "{:.1f}%\n$(n={:.0f})$".format(pct, absolute)

    def _update_axes_of_class_distribution_figure(
            self,
            fig: plt.Figure,
            axes: plt.axes,
            targets: np.ndarray,
            label_names: dict,
            show_legend: bool = False,
            title: Optional[str] = None
    ) -> None:
        """
        Updates axes of a pie chart with classes distribution.

        Parameters
        ----------
        fig : plt.Figure
            Figure.
        axes : plt.axes
            Axes.
        targets : np.ndarray
            Array of class targets.
        label_names : dict
            Dictionary with names associated to target values.
        show_legend : bool
            Whether to show legend.
        title : Optional[str]
            Title for the plot.
        """
        label_counts = {k: np.sum(targets == v) for k, v in label_names.items()}

        wedges, texts, auto_texts = axes.pie(
            label_counts.values(),
            textprops=dict(color="k"),
            startangle=90,
            autopct=lambda pct: self._format_to_percentage(pct, list(label_counts.values())),
            colors=self._light_colors,
            pctdistance=0.75,
            wedgeprops=dict(width=0.5)
        )

        for wedge in wedges:
            wedge.set_edgecolor("k")

        plt.setp(auto_texts, size=11)
        plt.setp(axes.collections, edgecolor="k")

        if show_legend:
            legend = fig.legend(
                wedges,
                list(label_counts.keys()),
                fontsize=14,
                handlelength=2.5,
                edgecolor="k",
                framealpha=1,
                loc="center",
                bbox_to_anchor=(0.5, 0.27)
            )
            for p in legend.get_patches():
                p.set_edgecolor(None)

        if title is not None:
            axes.set_xlabel(f"{title}\n$(n = {sum(label_counts.values())})$", fontsize=18, labelpad=-20)

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
        fig, axes = plt.subplots(figsize=self._fig_size, ncols=len(self._target_specific_masks[target]), squeeze=False)
        total = len(self._target_specific_masks[target])
        for idx, (name, mask) in enumerate(self._target_specific_masks[target].items()):
            if mask:
                filtered_df = self._target_specific_original_dfs[target].loc[
                    self._target_specific_original_dfs[target]["Sets"] == name
                    ]
                self._update_axes_of_class_distribution_figure(
                    fig=fig,
                    axes=axes[0, idx],
                    targets=filtered_df[target],
                    label_names={"Negative": 0, f"Positive": 1},
                    title=self._legend_names[name],
                    show_legend=idx == (total - 1)
                )
        terminate_figure(fig=fig, show=show, path_to_save=path_to_save)

    def _plot_continuous_feature_figure(
            self,
            dataframe: pd.DataFrame,
            feature: str,
            imputed: bool,
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
        imputed : bool
            Whether to use imputed data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        fig, axes = plt.subplots(figsize=self._fig_size)

        df = deepcopy(dataframe)
        value_counts = df['Sets'].value_counts()
        modified_legend_names = {k: f"{v}\n$(n={value_counts.get(k, 0)})$" for k, v in self._legend_names.items()}
        df['Sets'] = df['Sets'].replace(modified_legend_names)

        axes = sns.violinplot(
            data=df,
            y=feature,
            x="Sets",
            linewidth=1.3,
            inner="quart",
            saturation=1,
            cut=0
        )

        for i, line in enumerate(axes.get_lines()):
            line.set_color(self._dark_colors[i // 3])

        plt.setp(axes.collections, edgecolor="k")
        axes.minorticks_on()
        axes.set_ylabel(self._feature_names[feature] if self._feature_names else feature, fontsize=18)
        axes.tick_params(axis="y", direction='in', color="k", which="major", labelsize=16, length=6)
        axes.tick_params(axis="y", direction='in', color="k", which="minor", labelsize=16, length=3, bottom=False)
        axes.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=18)
        axes.xaxis.label.set_visible(False)

        if self._crop:
            if feature in self._crop.keys():
                crop_bounds = self._crop[feature][1] if imputed else self._crop[feature][0]
                upper_bound = crop_bounds[1]
                axes.set_ylim(crop_bounds[0], upper_bound)

                if upper_bound:
                    unique_values = df["Sets"].unique()

                    for i, v in enumerate(unique_values):
                        sliced_df = df[df["Sets"] == v]
                        values_higher_than_given = sliced_df[feature][sliced_df[feature] > upper_bound].sort_values()

                        if not values_higher_than_given.empty:
                            axes.annotate(
                                "",
                                xy=(i + 0.03, upper_bound),
                                xytext=(i + 0.14, upper_bound*0.86),
                                arrowprops=dict(
                                    arrowstyle="-|>, head_width=0.3",
                                    color="black",
                                    linewidth=1.3,
                                    connectionstyle="angle3,angleA=0,angleB=90"
                                )
                            )

                            axes.annotate(
                                f"Up to {round(values_higher_than_given.iloc[-1], 1)}",
                                xy=(i + 0.15, upper_bound*0.85),
                                xycoords="data",
                                xytext=(i + 0.15, upper_bound*0.85),
                                fontsize=16
                            )

        terminate_figure(fig=fig, show=show, path_to_save=path_to_save)

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
        self._plot_continuous_feature_figure(dataset, feature, imputed, path_to_save, show)

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
        self._plot_continuous_feature_figure(dataset, feature, imputed, path_to_save, show)

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
        logger = logging.getLogger('matplotlib')
        initial_logger_level = logger.getEffectiveLevel()
        logger.setLevel(logging.WARNING)

        df_copy = deepcopy(dataset)
        feature_series = df_copy[feature]

        fig, axes = plt.subplots(figsize=self._fig_size)

        feature_series_copy = deepcopy(feature_series.astype("category"))
        unique = pd.unique(feature_series_copy)
        if all(isinstance(x, (int, float)) for x in unique):
            axes.set_xticks(unique)

        if feature_series.dtype == "float":
            a, b, c = feature_series.min(), feature_series.max(), len(unique)
            bins = np.linspace(a - (b - a) / (2 * (c - 1)), b + (b - a) / (2 * (c - 1)), c + 1)

            sns.histplot(
                data=df_copy,
                x=feature,
                hue='Sets',
                bins=bins,
                multiple='dodge',
                ax=axes,
                stat='percent',
                common_norm=False,
                shrink=0.8,
                alpha=1,
                legend=False
            )

            fig_temp, axes_temp = plt.subplots()
            sns.histplot(
                data=df_copy,
                x=feature,
                bins=bins,
                hue='Sets',
                ax=axes_temp,
                alpha=1,
                legend=False
            )

        else:
            sns.histplot(
                data=df_copy,
                x=feature,
                hue='Sets',
                multiple='dodge',
                ax=axes,
                stat='percent',
                common_norm=False,
                shrink=0.8,
                alpha=1,
                legend=False
            )

            fig_temp, axes_temp = plt.subplots()
            sns.histplot(
                data=df_copy,
                x=feature,
                hue='Sets',
                ax=axes_temp,
                alpha=1,
                legend=False
            )

        for axes_container, axes_temp_container in zip(axes.containers, axes_temp.containers):
            labels = [f"$n={axes_temp_patch.get_height()}$" for axes_temp_patch in axes_temp_container]
            axes.bar_label(axes_container, labels=labels, fontsize=16)

        plt.close(fig_temp)
        axes.minorticks_on()
        axes.set_xlabel(self._feature_names[feature] if self._feature_names else feature, fontsize=18)
        axes.set_ylabel("Percentage $($%$)$", fontsize=18)
        axes.tick_params(axis="y", direction='in', color="k", which="major", labelsize=16, length=6)
        axes.tick_params(axis="y", direction='in', color="k", which="minor", labelsize=16, length=3, bottom=False)
        axes.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=16)

        handles = [plt.Line2D([0], [0], color=color, lw=8) for color in sns.color_palette()[:len(self._legend_names)]]
        labels = list(self._legend_names.values())
        axes.legend(handles=handles, labels=labels, loc="upper right", edgecolor="k", fontsize=16, handlelength=1.5)
        terminate_figure(fig=fig, show=show, path_to_save=path_to_save)

        logger.setLevel(initial_logger_level)

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

    @staticmethod
    def _build_kaplan_meier_curve(
            event_indicator: pd.Series,
            event_time: pd.Series,
            axes: plt.Axes,
            color: str,
            legend_label: Optional[str] = None,
            conf_level: float = 0.95,
    ) -> None:
        """
        Builds a Kaplan-Meier curve.

        Parameters
        ----------
        event_indicator : pd.Series
            Event indicator.
        event_time : pd.Series
            Event time.
        axes : plt.Axes
            Axes.
        color : str
            Color.
        legend_label : Optional[str]
            Legend label.
        conf_level : float
            Confidence level.
        """
        time, survival_probability, conf_int = kaplan_meier_estimator(
            event_indicator, event_time, conf_type="log-log", conf_level=conf_level
        )

        if legend_label is None:
            axes.step(time, survival_probability, where="post", color=color, lw=2)
        else:
            axes.step(time, survival_probability, where="post", color=color, lw=2, label=legend_label)

        axes.fill_between(time, conf_int[0], conf_int[1], alpha=0.3, step="post", color=color)

        censored_time = event_time[~event_indicator]
        censored_survival_probability = interp1d(time, survival_probability, kind="previous")(censored_time)
        axes.scatter(censored_time, censored_survival_probability, marker="|", s=100, color=color)

    @staticmethod
    def _get_structured_array(
            event_indicator: np.ndarray,
            event_time: np.ndarray
    ) -> np.ndarray:
        """
        Returns a structured array with event indicator and event time.

        Parameters
        ----------
        event_indicator : np.ndarray
            (N,) array with event indicator.
        event_time : np.ndarray
            (N,) array with event time.

        Returns
        -------
        structured_array : np.ndarray
            (N, 2) structured array with event indicator and event time.
        """
        structured_array = np.empty(shape=(len(event_indicator),), dtype=[('event', bool), ('time', float)])
        structured_array['event'] = event_indicator.astype(bool)
        structured_array['time'] = event_time

        return structured_array

    def _plot_masks_stratified_global_kaplan_meier_curve(
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
        fig, axes = plt.subplots(figsize=self._fig_size)

        survival_tables, colors = [], []
        for idx, (name, mask) in enumerate(self._global_masks.items()):
            subset = dataset.loc[dataset["Sets"] == name]
            event, time = subset[event_indicator].astype(bool), subset[event_time]
            self._build_kaplan_meier_curve(event, time, axes, self._light_colors[idx], name)

            survival_tables.append(survival_table_from_events(time, event))
            colors.append(self._light_colors[idx])

        add_details_to_kaplan_meier_curve(axes, True)
        add_at_risk_counts(survival_tables=survival_tables, colors=colors, axes=axes, figure=fig)

        try:
            filtered_df = dataset.dropna()
            _, p_value = compare_survival(
                y=self._get_structured_array(
                    event_indicator=filtered_df[event_indicator],
                    event_time=filtered_df[event_time]
                ),
                group_indicator=np.array([list(self._global_masks.keys()).index(row) for row in filtered_df["Sets"]])
            )

            axes.annotate(
                f"p-value$ = {p_value:.4f}$", xy=(0.15, 0.1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=16
            )
        except LinAlgError:
            pass

        texts = axes.get_legend().get_texts()
        for idx, (name, _) in enumerate(self._global_masks.items()):
            texts[idx].set_text(self._legend_names[name])

        terminate_figure(path_to_save=path_to_save, show=show, fig=fig)

    def _plot_unstratified_global_kaplan_meier_curve(
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
        fig, axes = plt.subplots(figsize=self._fig_size)

        event, time = dataset[event_indicator].astype(bool), dataset[event_time]
        self._build_kaplan_meier_curve(event, time, axes, self._light_colors[0])
        survival_table = survival_table_from_events(time, event)

        add_details_to_kaplan_meier_curve(axes, False)
        add_at_risk_counts(survival_tables=[survival_table], colors=[self._light_colors[0]], axes=axes, figure=fig)
        terminate_figure(path_to_save=path_to_save, show=show, fig=fig)

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
        dataframes = self._get_sets_dict(dataset)

        for name, df in dataframes.items():
            path_to_folder = os.path.join(path_to_save, name)
            os.makedirs(path_to_folder, exist_ok=True)

            path_to_fig = os.path.join(path_to_folder, "GLOBAL.png")
            self._plot_unstratified_global_kaplan_meier_curve(df, event_indicator, event_time, path_to_fig, show)

            if name == "full_dataset":
                path_to_fig = os.path.join(path_to_folder, "GLOBAL_STRATIFIED.png")
                self._plot_masks_stratified_global_kaplan_meier_curve(
                    dataset, event_indicator, event_time, path_to_fig, show
                )

    def _plot_categories_stratified_kaplan_meier_curve(
            self,
            dataset: pd.DataFrame,
            feature: str,
            event_indicator: str,
            event_time: str,
            path_to_save: Optional[str] = None,
            show: Optional[bool] = True
    ):
        """
        Plots the stratified kaplan meier curve for each category of the feature.

        Parameters
        ----------
        dataset : pd.DataFrame
            Dataframe.
        feature : str
            The name of the column containing the feature data.
        event_indicator : str
            The name of the column containing the event indicator data.
        event_time : str
            The name of the column containing the event time data.
        path_to_save : Optional[str]
            Path to save descriptive analysis records.
        show : Optional[bool]
            Whether to show figures.
        """
        fig, axes = plt.subplots(figsize=self._fig_size)
        unique = pd.unique(dataset[feature])
        values_count = dataset[feature].value_counts(sort=False)
        unique_categories = [category for category in unique.categories if values_count[category] >= 1]
        survival_tables, colors = [], []
        for n, category in enumerate(unique_categories):
            mask = dataset[feature] == category
            event, time = dataset[event_indicator][mask].astype(bool), dataset[event_time][mask]

            self._build_kaplan_meier_curve(event, time, axes, self._light_colors[n], category)
            survival_tables.append(survival_table_from_events(time, event))
            colors.append(self._light_colors[n])

        add_details_to_kaplan_meier_curve(axes, True)
        add_at_risk_counts(survival_tables=survival_tables, colors=colors, axes=axes, figure=fig)

        try:
            filtered_df = dataset.dropna()
            _, p_value = compare_survival(
                y=self._get_structured_array(
                    event_indicator=filtered_df[event_indicator],
                    event_time=filtered_df[event_time]
                ),
                group_indicator=np.array([unique_categories.index(row) for row in filtered_df[feature].dropna()])
            )

            axes.annotate(
                f"p-value$ = {p_value:.4f}$", xy=(0.15, 0.1), xycoords="axes fraction", textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=16
            )
        except LinAlgError:
            pass

        legend = axes.get_legend()
        title = self._feature_names[feature]
        second_word = title.split()[1]
        title = title.replace(" ", "\n") if second_word[0].isupper() else title
        legend.set_title(title, prop={"size": 16})
        plt.setp(legend.get_title(), multialignment='center')

        terminate_figure(fig=fig, show=show, path_to_save=path_to_save)

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

        dataframes = self._get_sets_dict(df_copy)
        for name, df in dataframes.items():
            path_to_folder = os.path.join(path_to_save, name)
            os.makedirs(path_to_folder, exist_ok=True)

            path_to_fig = os.path.join(path_to_folder, f"{feature}.png")
            self._plot_categories_stratified_kaplan_meier_curve(
                df, feature, event_indicator, event_time, path_to_fig, show
            )

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

        os.makedirs(os.path.join(figures_path, f"{self.TARGET_PATH}s"), exist_ok=True)

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

    def _save_tables(
            self,
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

        dataframes = self._get_sets_dict(dataframe)
        for name, df in dataframes.items():
            path_to_folder = os.path.join(path_to_save, name)
            os.makedirs(path_to_folder, exist_ok=True)

            df.describe().transpose().round(1).to_csv(
                os.path.join(path_to_folder, f"description_cont_features.csv"), index=True
            )

            target_df = pd.DataFrame(columns=["Target", "Mean", "Std"])
            for task in self.dataset.tasks.survival_analysis_tasks:
                event_column, time_column = task.event_indicator_column, task.event_time_column
                filtered_df = df[df[event_column] == 1][time_column]
                target_df = pd.concat(
                    [target_df, pd.DataFrame(
                        {
                            "Target": event_column,
                            "Mean": [filtered_df.mean()],
                            "Median": [filtered_df.median()],
                            "Min": [filtered_df.min()],
                            "Max": [filtered_df.max()],
                            "Std": [filtered_df.std()],
                            "Follow-up T-OBS": [df[time_column].median()],
                            "Follow-up T-CENS": [df[df[event_column] == 0][time_column].median()]
                        })
                     ],
                    ignore_index=True
                )
            target_df.to_csv(os.path.join(path_to_folder, f"description_target.csv"), index=False)

            frequency_table = self.get_frequency_table(df)
            frequency_table.to_csv(os.path.join(path_to_folder, f"description_cat_features.csv"), index=False)

    def get_frequency_table(
            self,
            dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Returns a frequency table.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.

        Returns
        -------
        frequency_table : pd.DataFrame
            Frequency table.
        """
        dataframes = []
        for column_idx, column_name in enumerate(self.dataset.categorical_features_columns):
            frequency_table = self._get_count_and_percentage_dataframe(column_name, dataframe)

            frequency_table = self._get_frequency_table_with_concatenated_list(
                frequency_table=frequency_table,
                values=list(frequency_table.index),
                first_column=True
            )

            number_of_levels = len(frequency_table.index)
            variable = [""] * number_of_levels
            variable[0] = column_name
            frequency_table = self._get_frequency_table_with_concatenated_list(
                frequency_table=frequency_table,
                values=variable,
                first_column=True
            )

            dataframes.append(frequency_table)

        dataframe = pd.concat(dataframes)
        columns = ["Variable", "Level", "n", "%"]
        dataframe.columns = columns

        return dataframe

    def _save_target_specific_continuous_tables(
            self,
            dataframe: pd.DataFrame,
            target: str,
            path_to_save: str
    ) -> None:
        """
        Saves dataframes in the .csv format.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        target : str
            Target column.
        path_to_save : str
            Path to save tables.
        """
        dataframes = self._get_sets_dict(dataframe)

        for name, df in dataframes.items():
            negative_df = df[df[target] == 0][[target] + self.dataset.continuous_features_columns]
            positive_df = df[df[target] == 1][[target] + self.dataset.continuous_features_columns]

            negative_stats_df = negative_df.describe().transpose().round(1).reset_index()
            positive_stats_df = positive_df.describe().transpose().round(1).reset_index()

            negative_stats_df.insert(0, "Level", 0)
            positive_stats_df.insert(0, "Level", 1)

            concat_df = pd.concat([negative_stats_df, positive_stats_df]).sort_index().set_index("index")

            concat_df.index = ["" if idx % 2 != 0 else label for idx, label in enumerate(concat_df.index)]
            concat_df.insert(0, "Variable", concat_df.index)

            p_values = []
            for idx, label in enumerate(concat_df.index):
                if idx % 2 != 0:
                    p_value = ""
                else:
                    p_value = self._get_p_value_from_mann_whitney_u_test(
                        column_name=label,
                        negative_outcome_dataframe=negative_df,
                        positive_outcome_dataframe=positive_df
                    )
                p_values.append(p_value)

            concat_df["p-value"] = p_values

            path_to_folder = os.path.join(path_to_save, name)
            os.makedirs(path_to_folder, exist_ok=True)

            concat_df.to_csv(os.path.join(path_to_folder, f"target_description_cont_features.csv"), index=False)

    @staticmethod
    def _get_p_value_from_mann_whitney_u_test(
            column_name: str,
            negative_outcome_dataframe: pd.DataFrame,
            positive_outcome_dataframe: pd.DataFrame
    ) -> float:
        """
        Calculates p-value from Mann-Whitney U test.

        Parameters
        ----------
        column_name : str
            Column name.
        negative_outcome_dataframe : pd.DataFrame
            Negative outcome dataframe.
        positive_outcome_dataframe : pd.DataFrame
            Positive outcome dataframe.

        Returns
        -------
        p-value : float
            P-value.
        """
        _, p_value = mannwhitneyu(
            x=negative_outcome_dataframe[column_name].dropna(),
            y=positive_outcome_dataframe[column_name].dropna()
        )

        return p_value

    def _save_target_specific_categorical_tables(
            self,
            dataframe: pd.DataFrame,
            target: str,
            path_to_save: str
    ) -> None:
        """
        Saves dataframes in the .csv format.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe.
        target : str
            Target column.
        path_to_save : str
            Path to save tables.
        """
        dataframes = self._get_sets_dict(dataframe)

        for name, df in dataframes.items():
            dataframes = []
            for column_idx, column_name in enumerate(self.dataset.categorical_features_columns):
                negative_df, positive_df = df[df[target] == 0], df[df[target] == 1]
                frequency_table = self._get_count_and_percentage_dataframe(column_name, negative_df, positive_df)
                number_of_levels = len(frequency_table.index)

                p_value = [""] * number_of_levels
                p_value[0] = str(
                    round(
                        self._get_p_value_from_chi2_test_on_frequency_table(
                            column_name=column_name, negative_df=negative_df, positive_df=positive_df
                        ),
                        ndigits=4
                    )
                )
                frequency_table = self._get_frequency_table_with_concatenated_list(
                    frequency_table=frequency_table,
                    values=p_value
                )

                frequency_table = self._get_frequency_table_with_concatenated_list(
                    frequency_table=frequency_table,
                    values=list(frequency_table.index),
                    first_column=True
                )

                variable = [""] * number_of_levels
                variable[0] = column_name
                frequency_table = self._get_frequency_table_with_concatenated_list(
                    frequency_table=frequency_table,
                    values=variable,
                    first_column=True
                )

                frequency_table = frequency_table[[0, 1, 2, 4, 3, 5, 6]]

                dataframes.append(frequency_table)

            dataframe_ = pd.concat(dataframes)
            columns = ["Variable", "Level", "Negative n/N", "Negative %", "Positive n/N", "Positive %", "p-value"]
            dataframe_.columns = columns

            path_to_folder = os.path.join(path_to_save, name)
            os.makedirs(path_to_folder, exist_ok=True)

            dataframe_.to_csv(os.path.join(path_to_folder, f"target_description_cat_features.csv"), index=False)

    def _get_count_and_percentage_dataframe(
            self,
            column_name: str,
            negative_df: pd.DataFrame,
            positive_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Returns count and percentage dataframe.

        Parameters
        ----------
        column_name : str
            Column name.
        negative_df : pd.DataFrame
            Negative dataframe.
        positive_df : Optional[pd.DataFrame]
            Positive dataframe.

        Returns
        -------
        count_and_percentage_dataframe : pd.DataFrame
            Count and percentage dataframe.
        """
        count_and_percentage_dataframe = pd.merge(
            left=self._get_count_dataframe(column_name, negative_df, positive_df),
            right=self._get_percentage_dataframe(column_name, negative_df, positive_df),
            left_index=True,
            right_index=True
        )

        return count_and_percentage_dataframe

    @staticmethod
    def _get_count_dataframe(
            column_name: str,
            negative_df: pd.DataFrame,
            positive_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Returns count dataframe.

        Parameters
        ----------
        column_name : str
            Column name.
        negative_df : pd.DataFrame
            Negative dataframe.
        positive_df : Optional[pd.DataFrame]
            Positive dataframe.

        Returns
        -------
        count_dataframe : pd.DataFrame
            Count dataframe.
        """
        if positive_df is not None:
            data = [
                negative_df[column_name].value_counts(sort=False),
                positive_df[column_name].value_counts(sort=False)
            ]

            count_dataframe_int: pd.DataFrame(dtype=int) = pd.concat(data, axis=1).fillna(0).applymap(int)
            count_dataframe_str: pd.DataFrame(dtype=str) = count_dataframe_int.applymap(str)

            for column_idx, _ in enumerate(count_dataframe_int.columns):
                column_sum = count_dataframe_int.iloc[:, column_idx].sum()
                count_dataframe_str.iloc[:, column_idx] = count_dataframe_str.iloc[:, column_idx] + f"//{column_sum}"
        else:
            count_dataframe_int = negative_df[column_name].value_counts(sort=False).fillna(0).apply(int)
            count_dataframe_str: pd.DataFrame(dtype=str) = count_dataframe_int.apply(str)

            column_sum = count_dataframe_int.sum()
            count_dataframe_str = count_dataframe_str + f"//{column_sum}"

        return count_dataframe_str

    def _get_p_value_from_chi2_test_on_frequency_table(
            self,
            column_name: str,
            negative_df: pd.DataFrame,
            positive_df: pd.DataFrame
    ) -> float:
        """
        Calculates p-value from chi2 test on frequency table.

        Parameters
        ----------
        column_name : str
            Column name.
        negative_df : pd.DataFrame
            Negative dataframe.
        positive_df : pd.DataFrame
            Positive dataframe.

        Returns
        -------
        p-value : float
            P-value.
        """
        result = pd.concat(
            [
                negative_df[column_name].value_counts(sort=False),
                positive_df[column_name].value_counts(sort=False)
            ], axis=1
        ).fillna(0)
        result = result.loc[~(result == 0).all(axis=1)]

        chi2, p_value, dof, expected = chi2_contingency(observed=result)

        return p_value

    @staticmethod
    def _get_frequency_table_with_concatenated_list(
            frequency_table: pd.DataFrame,
            values: list,
            first_column: bool = False
    ) -> pd.DataFrame:
        """
        Returns frequency table with concatenated list.

        Parameters
        ----------
        frequency_table : pd.DataFrame
            Frequency table.
        values : list
            Values.
        first_column : bool
            If True, then values will be concatenated to the first column.

        Returns
        -------
        frequency_table : pd.DataFrame
            Frequency table.
        """
        series = pd.Series(data=values, index=frequency_table.index)

        if first_column:
            data = [series, frequency_table]
        else:
            data = [frequency_table, series]

        frequency_table = pd.concat(data, axis=1, ignore_index=True)

        return frequency_table

    @staticmethod
    def _get_percentage_dataframe(
            column_name: str,
            negative_df: pd.DataFrame,
            positive_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Returns percentage dataframe.

        Parameters
        ----------
        column_name : str
            Column name.
        negative_df : pd.DataFrame
            Negative dataframe.
        positive_df : pd.DataFrame
            Positive dataframe.

        Returns
        -------
        percentage_dataframe : pd.DataFrame
            Percentage dataframe.
        """
        if positive_df is not None:
            data = [
                round(negative_df[column_name].value_counts(sort=False, normalize=True)*100, ndigits=1),
                round(positive_df[column_name].value_counts(sort=False, normalize=True)*100, ndigits=1)
            ]

            percentage_dataframe: pd.DataFrame(dtype=int) = pd.concat(data, axis=1).fillna(0)
        else:
            percentage_dataframe = round(
                negative_df[column_name].value_counts(sort=False, normalize=True) * 100, ndigits=1
            )

        return percentage_dataframe

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
                self._save_target_specific_continuous_tables(df, target, path_to_save_tables)
                self._save_target_specific_categorical_tables(df, target, path_to_save_tables)

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
        for imputed, path in [(False, self.ORIGINAL_DF_PATH), (True, self.IMPUTED_DF_PATH)]:
            path_to_folder = os.path.join(
                path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, self.CORRELATIONS_PATH, path
            )
            self.visualize_correlations(ds.target_columns, imputed, path_to_folder, False)

    def _save_timeline_figures(
            self,
            path_to_save: str
    ) -> None:
        """
        Saves timeline figures.

        Parameters
        ----------
        path_to_save : str
            Path to save timeline figures.
        """
        dataframes = self._get_sets_dict(self._global_original_df)

        for name, df in dataframes.items():
            path_to_folder = os.path.join(path_to_save, self.GLOBAL_PATH, self.FIGURES_PATH, f"{self.TARGET_PATH}s")
            self.visualize_timeline(name, df, path_to_folder, False)

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
            for cont_col in self.dataset.continuous_features_columns:
                path_to_fig = os.path.join(directory, f"{cont_col}.png")
                self.visualize_global_continuous_feature(cont_col, imputed, path_to_fig, False)
            for cat_col in self.dataset.categorical_features_columns:
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
                for cont_col in self.dataset.continuous_features_columns:
                    path_to_fig = os.path.join(directory, f"{cont_col}.png")
                    self.visualize_target_specific_continuous_features(cont_col, target, imputed, path_to_fig, False)
                for cat_col in self.dataset.categorical_features_columns:
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
                self.visualize_kaplan_meier_curve(event, time, imputed, None, directory, False)

                for cat_col in self.dataset.categorical_features_columns:
                    self.visualize_kaplan_meier_curve(event, time, imputed, cat_col, directory, False)

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
        self._save_timeline_figures(path_to_save)
        self._save_targets_figures(path_to_save)
        self._save_features_figures(path_to_save)
        self._save_kaplan_meier_figures(path_to_save)
