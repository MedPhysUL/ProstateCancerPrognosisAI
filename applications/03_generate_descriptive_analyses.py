"""
    @file:              03_generate_descriptive_analyses.py
    @Author:            Maxence Larose

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Script used to generate a folder containing descriptive analyses of all data.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # ----------------------------------------------------------------------------------------------------------- #
    #                                                DataFrames                                                   #
    # ----------------------------------------------------------------------------------------------------------- #
    learning_df = pd.DataFrame(
        [[1, 0, "1"], [1.1, 0, "1"], [0.9, 0, "0"], [1, 0, "0"], [1.2, 0, "0"], [1.03, 0, "2"]],
        columns=["A", "B", "C"]
    )
    holdout_df = pd.DataFrame(
        [[2, 0, "2"], [2.1, 0, "2"], [1.9, 0, "1"], [2, 0, "1"], [2.2, 0, "2"], [2.03, 0, "0"]],
        columns=["A", "B", "C"]
    )

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
        y="A",
        x="Set",
        linewidth=1,
    )
    plt.show()

    fig, ax = plt.subplots()
    sns.histplot(
        data=all_set,
        x='C', hue='Set',
        multiple='dodge',
        ax=ax
    )

    plt.show()
