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

from constants import (
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    EXPERIMENTS_PATH,
    FINAL_TABLE_PATH,
    MASKS_PATH,
    ID,
    SEED,
    BCR_TASK,
    PN_TASK,
    METASTASIS_TASK,
    HTX_TASK,
    CRPC_TASK,
    DEATH_TASK
)
from src.data.datasets import Feature
from src.data.transforms import Normalization
from src.data.processing.sampling import Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.models.torch.prediction import SequentialNet, MLP


if __name__ == '__main__':
    split = 4
    df = pd.read_csv(
        fr"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\radiomics\multitask\outer_split_{split}\outer_split.csv"
    )

    RADIOMIC_7 = Feature(column="RADIOMIC_BCR_1", transform=Normalization(), impute=False)
    RADIOMIC_8 = Feature(column="RADIOMIC_BCR_2", transform=Normalization(), impute=False)
    RADIOMIC_9 = Feature(column="RADIOMIC_BCR_3", transform=Normalization(), impute=False)
    RADIOMIC_10 = Feature(column="RADIOMIC_BCR_4", transform=Normalization(), impute=False)
    RADIOMIC_11 = Feature(column="RADIOMIC_BCR_5", transform=Normalization(), impute=False)
    RADIOMIC_12 = Feature(column="RADIOMIC_BCR_6", transform=Normalization(), impute=False)

    BCR_RADIOMICS = [RADIOMIC_7, RADIOMIC_8, RADIOMIC_9, RADIOMIC_10, RADIOMIC_11, RADIOMIC_12]

    RADIOMICS = BCR_RADIOMICS

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=[BCR_TASK],
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
    model = MLP(
        n_layers=1,
        n_neurons=20,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\BCR(MLP - Clinical data and deep radiomics)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
    )
    model.load_state_dict(state)

    model.fix_thresholds_to_optimal_values(dataset)
    model.fit_breslow_estimators(dataset)

    # history.plot(show=True)

    score = model.compute_score_on_dataset(dataset, dataset.train_mask)
    print(score)
    score = model.compute_score_on_dataset(dataset, dataset.valid_mask)
    print(score)
    score = model.compute_score_on_dataset(dataset, dataset.test_mask, n_samples=100)
    print(score)

    preds = model.predict_on_dataset(dataset, dataset.test_mask, n_samples=100)
    preds = {k: torch.transpose(v, 1, 0).tolist()[0] for k, v in preds.items()}
    import json
    path = r'C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\preds\split_4\BCR_CD_DR.json'
    with open(path, 'w') as file:
        json.dump(preds, file)
    # evaluator.plot_binary_classification_task_curves(show=True)
    # evaluator.plot_survival_analysis_task_curves(show=True)
