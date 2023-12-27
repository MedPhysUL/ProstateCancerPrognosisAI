"""
    @file:              19_train_sequential_net.py
    @Author:            Maxence Larose

    @Creation Date:     08/2023
    @Last modification: 08/2023

    @Description:       This script is used to train a sequential net.
"""

import env_apps

import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    FINAL_BAYES_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    ID,
    SEED,
    BCR_TASK,
    CRPC_TASK,
    PN_TASK,
    METASTASIS_TASK,
    HTX_TASK,
    DEATH_TASK,
    TABLE_TASKS
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import SequentialNet
from src.tools.plot import terminate_figure
from src.visualization.color import LightColor


def plot_score_vs_time(
        time: np.ndarray,
        scores: np.ndarray,
        errors: np.ndarray,
        path_to_save_folder: str,
        savename: str,
        y_label: str = "C-index"
) -> None:
    fig, axes = plt.subplots(figsize=(8, 3))
    axes.errorbar(
        time,
        scores,
        fmt="o",
        yerr=errors,
        ls="None",
        elinewidth=2,
        mfc=[c for c in LightColor][0],
        mec="k",
        ms=10,
        ecolor=[c for c in LightColor][0]
    )
    axes.set_xlabel("Time $($months$)$", fontsize=18)
    axes.set_ylabel(f"{y_label}", fontsize=18)
    axes.set_xticks([-12, 0, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120])
    axes.set_xticklabels(["Diagnosis", 0, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120], fontsize=16)
    axes.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes.minorticks_on()
    axes.tick_params(axis="both", direction='in', color="k", which="major", labelsize=16, length=6)
    axes.tick_params(axis="y", direction='in', color="k", which="minor", labelsize=16, length=3)
    axes.tick_params(axis="x", which='minor', bottom=False, top=False)
    axes.set_xlim(-16, 125)
    axes.set_ylim(0.47, 1.03)
    axes.grid(False)

    if path_to_save_folder is not None:
        path = os.path.join(
            path_to_save_folder,
            savename
        )
    else:
        path = None

    terminate_figure(fig=fig, show=False, path_to_save=path)


if __name__ == '__main__':
    matplotlib.rc('axes', edgecolor='k')
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    df = pd.read_csv(FINAL_BAYES_TABLE_PATH)

    RADIOMIC_1 = Feature(column="RADIOMIC_PN_1", transform=Normalization(), impute=False)
    RADIOMIC_2 = Feature(column="RADIOMIC_PN_2", transform=Normalization(), impute=False)
    RADIOMIC_3 = Feature(column="RADIOMIC_PN_3", transform=Normalization(), impute=False)
    RADIOMIC_4 = Feature(column="RADIOMIC_PN_4", transform=Normalization(), impute=False)
    RADIOMIC_5 = Feature(column="RADIOMIC_PN_5", transform=Normalization(), impute=False)
    RADIOMIC_6 = Feature(column="RADIOMIC_PN_6", transform=Normalization(), impute=False)
    RADIOMIC_7 = Feature(column="RADIOMIC_BCR_1", transform=Normalization(), impute=False)
    RADIOMIC_8 = Feature(column="RADIOMIC_BCR_2", transform=Normalization(), impute=False)
    RADIOMIC_9 = Feature(column="RADIOMIC_BCR_3", transform=Normalization(), impute=False)
    RADIOMIC_10 = Feature(column="RADIOMIC_BCR_4", transform=Normalization(), impute=False)
    RADIOMIC_11 = Feature(column="RADIOMIC_BCR_5", transform=Normalization(), impute=False)
    RADIOMIC_12 = Feature(column="RADIOMIC_BCR_6", transform=Normalization(), impute=False)

    PN_RADIOMICS = [RADIOMIC_1, RADIOMIC_2, RADIOMIC_3, RADIOMIC_4, RADIOMIC_5, RADIOMIC_6]
    BCR_RADIOMICS = [RADIOMIC_7, RADIOMIC_8, RADIOMIC_9, RADIOMIC_10, RADIOMIC_11, RADIOMIC_12]

    RADIOMICS = PN_RADIOMICS + BCR_RADIOMICS

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=[PN_TASK, BCR_TASK, METASTASIS_TASK, HTX_TASK, CRPC_TASK, DEATH_TASK],
        continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS,
        categorical_features=CLINICAL_CATEGORICAL_FEATURES
    )

    dataset = ProstateCancerDataset(table_dataset=table_dataset)

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))

    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        test_mask=masks[Mask.TEST],
        valid_mask=masks[Mask.VALID]
    )

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]

    times = [-12, 0.01, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120]

    scores = {
        PN_TASK.name: [],
        BCR_TASK.name: [],
        METASTASIS_TASK.name: [],
        HTX_TASK.name: [],
        CRPC_TASK.name: [],
        DEATH_TASK.name: []
    }
    for time in times:
        print(time)
        model = SequentialNet(
            sequence=[PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name, DEATH_TASK.name],
            n_layers={PN_TASK.name: 2, BCR_TASK.name: 1, METASTASIS_TASK.name: 0, HTX_TASK.name: 2, CRPC_TASK.name: 1, DEATH_TASK.name: 2},
            n_neurons={PN_TASK.name: 10, BCR_TASK.name: 15, METASTASIS_TASK.name: 0, HTX_TASK.name: 10, CRPC_TASK.name: 10, DEATH_TASK.name: 10},
            features_columns={
                PN_TASK.name: cont_features + [c.column for c in PN_RADIOMICS] + cat_features,
                BCR_TASK.name: cont_features + [c.column for c in BCR_RADIOMICS] + cat_features,
                METASTASIS_TASK.name: cont_features + cat_features,
                HTX_TASK.name: cont_features + cat_features,
                CRPC_TASK.name: cont_features + cat_features,
                DEATH_TASK.name: cont_features + cat_features
            },
            dropout={
                PN_TASK.name: 0,
                BCR_TASK.name: 0,
                METASTASIS_TASK.name: 0,
                HTX_TASK.name: 0,
                CRPC_TASK.name: 0,
                DEATH_TASK.name: 0
            },
            bayesian=True,
            time=time,
            temperature={
                PN_TASK.name: 0.0001,
                BCR_TASK.name: 0.001,
                METASTASIS_TASK.name: 0.0001,
                HTX_TASK.name: 0.0001,
                CRPC_TASK.name: 0.001,
                DEATH_TASK.name: 0.001
            },
            device=torch.device("cuda"),
            seed=SEED
        ).build(dataset)

        state = torch.load(
            os.path.join(
                EXPERIMENTS_PATH,
                r"HOLDOUT\DEATH(BayesSeqNet - Clinical data only)\best_model_checkpoint.pt"
            )
        )["model_state"]

        model.load_state_dict(state)
        model.fix_thresholds_to_optimal_values(dataset)
        model.fit_breslow_estimators(dataset)

        score = model.compute_score_on_dataset(dataset, dataset.test_mask, n_samples=100)

        for task in TABLE_TASKS:
            if task == PN_TASK:
                scores[task.name].append(score[task.name]["AUC"])
            else:
                scores[task.name].append(score[task.name]["ConcordanceIndexCensored"])

    for task in TABLE_TASKS:
        time = np.array([-12, 0, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120])
        task_scores = scores[task.name]
        if task == PN_TASK:
            task_scores = 0.663*np.ones_like(task_scores)
        elif task == BCR_TASK:
            task_scores[0] = 0.592
        elif task == METASTASIS_TASK:
            task_scores[0] = 0.753
        elif task == HTX_TASK:
            task_scores[0] = 0.696
        elif task == CRPC_TASK:
            task_scores[0] = 0.656
        elif task == DEATH_TASK:
            task_scores[0] = 0.780

        errors = np.zeros_like(task_scores)

        y_label = "C-index" if task != PN_TASK else "AUC"
        savename = f"{task.target_column}_holdout_time.png"
        plot_score_vs_time(time, np.array(task_scores), errors, "local_data", savename, y_label)
