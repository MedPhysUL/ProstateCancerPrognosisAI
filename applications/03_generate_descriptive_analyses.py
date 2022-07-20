"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Script used to generate a folder containing descriptive analyses of all data.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def format_to_percentage(
        pct: float, values: List[float]
) -> str:
    """
    Change a float to a str representing a percentage
    Args:
        pct: count related to a class
        values: count of items in each class
    Returns: str
    """
    absolute = int(round(pct/100.*np.sum(values)))
    return "{:.1f}%".format(pct, absolute)


def visualize_class_distribution(
        targets: np.array,
        label_names: dict,
        title: Optional[str] = None
) -> None:
    """
    Shows a pie chart with classes distribution
    Args:
        targets: array of class targets
        label_names: dictionary with names associated to target values
        title: title for the plot
    Returns: None
    """
    # We first count the number of instances of each value in the targets vector
    label_counts = {k: np.sum(targets == v) for k, v in label_names.items()}

    # We prepare a list of string to use as plot labels
    labels = [f"{k} ({v})" for k, v in label_counts.items()]

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    fig, ax = plt.subplots()

    wedges, texts, autotexts = ax.pie(
        label_counts.values(),
        textprops=dict(color="w"),
        startangle=90,
        autopct=lambda pct: format_to_percentage(pct, list(label_counts.values()))
    )

    ax.legend(
        wedges,
        labels,
        title="Labels",
        loc="center right",
        bbox_to_anchor=(0.1, 0.5, 0, 0),
        prop={"size": 8}
    )

    plt.setp(autotexts, size=8, weight="bold")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.read_csv("local_data/learning_table.csv")
    holdout_df = pd.read_csv("local_data/holdout_table.csv")

    # ----------------------------------------------------------------------------------------------------------- #
    #                                            Descriptive analysis                                             #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df.describe()
    holdout_df.describe()

    learning_df["Set"] = ["Learning"]*len(learning_df)
    holdout_df["Set"] = ["Holdout"] * len(holdout_df)

    all_set = pd.concat([learning_df, holdout_df], ignore_index=True)

    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=all_set,
        y="PSA",
        x="Set",
        linewidth=1,
    )
    plt.show()

    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=all_set,
        y="AGE",
        x="Set",
        linewidth=1,
    )
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(
        data=all_set,
        x='CLINICAL_STAGE', hue='Set',
        multiple='dodge',
        ax=ax,
        stat='probability',
        common_norm=False,
        shrink=0.8
    )
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(
        data=all_set,
        x='GLEASON_GLOBAL', hue='Set',
        multiple='dodge',
        ax=ax,
        stat='probability',
        common_norm=False,
        shrink=1.5
    )
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(
        data=all_set,
        x='GLEASON_PRIMARY', hue='Set',
        multiple='dodge',
        ax=ax,
        stat='probability',
        common_norm=False,
        shrink=1.5
    )
    plt.show()

    visualize_class_distribution(targets=np.array(learning_df["PN"]), label_names={"PN0": 0, "PN1": 1})
    visualize_class_distribution(targets=np.array(learning_df["BCR"]), label_names={"No recurrence": 0, "Recurrence": 1})

    visualize_class_distribution(targets=np.array(holdout_df["PN"]), label_names={"PN0": 0, "PN1": 1})
    visualize_class_distribution(targets=np.array(holdout_df["BCR"]), label_names={"No recurrence": 0, "Recurrence": 1})
