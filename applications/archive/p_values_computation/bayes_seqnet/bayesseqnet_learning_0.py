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

import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    FINAL_BAYES_TABLE_PATH,
    HOLDOUT_MASKS_PATH,
    MASKS_PATH,
    ID,
    SEED,
    BCR_TASK,
    CRPC_TASK,
    PN_TASK,
    METASTASIS_TASK,
    HTX_TASK,
    DEATH_TASK
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.models.torch.prediction import SequentialNet, ModelConfig
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    split = 0
    df = pd.read_csv(
        fr"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\radiomics\multitask_bayesian\outer_split_{split}\outer_split.csv"
    )

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

    masks = json.load(open(MASKS_PATH, "r"))

    dataset.update_masks(
        train_mask=masks[f"{split}"][Mask.TRAIN],
        test_mask=masks[f"{split}"][Mask.TEST],
        valid_mask=masks[f"{split}"][Mask.VALID]
    )

    cont_features = [c.column for c in CLINICAL_CONTINUOUS_FEATURES]
    cat_features = [c.column for c in CLINICAL_CATEGORICAL_FEATURES]
    model = SequentialNet(
        sequence=[PN_TASK.name, BCR_TASK.name, METASTASIS_TASK.name, HTX_TASK.name, CRPC_TASK.name, DEATH_TASK.name],
        n_layers={PN_TASK.name: 3, BCR_TASK.name: 3, METASTASIS_TASK.name: 2, HTX_TASK.name: 2, CRPC_TASK.name: 2, DEATH_TASK.name: 3},
        n_neurons={PN_TASK.name: 20, BCR_TASK.name: 20, METASTASIS_TASK.name: 20, HTX_TASK.name: 20, CRPC_TASK.name: 10, DEATH_TASK.name: 15},
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
            DEATH_TASK.name: 0.05
        },
        time=60,
        bayesian=True,
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
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\DEATH(BayesSeqNet - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
    )
    model.load_state_dict(state)
    model.fix_thresholds_to_optimal_values(dataset)
    model.fit_breslow_estimators(dataset)

    # history.plot(show=True)
    score = model.compute_score_on_dataset(dataset, dataset.test_mask, n_samples=100)
    print(score)

    # evaluator.plot_binary_classification_task_curves(show=True)
    # evaluator.plot_survival_analysis_task_curves(show=True)
