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
    #####################################################
    predictions = {}
    split = 4

    df = pd.read_csv(
        fr"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\radiomics\multitask\outer_split_{split}\outer_split.csv"
    )

    task = PN_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=3,
        n_neurons=20,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]

    ######################

    task = BCR_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=1,
        n_neurons=15,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]


    ######################

    task = CRPC_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=1,
        n_neurons=20,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]

    ######################

    task = HTX_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=3,
        n_neurons=20,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]

    ######################

    task = METASTASIS_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=2,
        n_neurons=20,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]

    ######################

    task = DEATH_TASK
    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=task,
        continuous_features=CLINICAL_CONTINUOUS_FEATURES,
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
        features_columns=cont_features + cat_features,
        n_layers=3,
        n_neurons=5,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    state = torch.load(
        rf"C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\records\experiments\{task.target_column}(MLP - Clinical data only)\outer_splits\split_{split}\best_models\outer_split\best_model.pt"
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
    predictions[task.name] = preds[task.name]

    import json
    path = r'C:\Users\maxen\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\preds\split_4\MlP_clinical_data.json'
    with open(path, 'w') as file:
        json.dump(predictions, file)
