import env_apps

import json
import numpy as np
import os
import pandas as pd
import torch

from constants import (
    AUTOMATIC_FILTERED_RADIOMICS_PATH,
    CLINICAL_CATEGORICAL_FEATURES,
    CLINICAL_CONTINUOUS_FEATURES,
    RADIOMICS_FEATURES,
    ID,
    LEARNING_TABLE_PATH,
    MANUAL_FILTERED_RADIOMICS_PATH,
    MASKS_PATH,
    SEED,
    TABLE_TASKS
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ProstateCancerDataset, TableDataset
from src.evaluation import ModelEvaluator
from src.models.torch.prediction import MLP


if __name__ == '__main__':
    df = pd.read_csv(LEARNING_TABLE_PATH)

    for task in TABLE_TASKS:
        table_dataset = TableDataset(
            dataframe=pd.read_csv(
                    os.path.join(
                        MANUAL_FILTERED_RADIOMICS_PATH, task.target_column, f"outer_split_0", "outer_split.csv"
                    )
                ),
            ids_column=ID,
            tasks=task,
            continuous_features=CLINICAL_CONTINUOUS_FEATURES + RADIOMICS_FEATURES,
            categorical_features=CLINICAL_CATEGORICAL_FEATURES
        )

        dataset = ProstateCancerDataset(table_dataset=table_dataset)

        masks = extract_masks(MASKS_PATH, k=5, l=5)
        c_index, c_index_ipcw, cda = [], [], []
        for i in range(5):
            dataset.update_dataframe(
                dataframe=pd.read_csv(
                    os.path.join(
                        MANUAL_FILTERED_RADIOMICS_PATH, task.target_column, f"outer_split_{i}", "outer_split.csv"
                    )
                ),
                update_masks=False
            )

            dataset.update_masks(
                train_mask=masks[i][Mask.TRAIN],
                test_mask=masks[i][Mask.TEST],
                valid_mask=masks[i][Mask.VALID]
            )

            file = open(
                f"local_data/records/experiments/{task.target_column}(MLP - Clinical data and manual radiomics)/"
                f"outer_splits/split_{i}/best_hyperparameters.json"
            )
            hidden_channels = json.load(file)["best_trial"]["_params"]["hidden_channels"]

            model = MLP(
                activation="PRELU",
                dropout=0.2,
                hidden_channels=hidden_channels,
                device=torch.device("cuda"),
                seed=SEED
            ).build(dataset)

            model.load_state_dict(
                torch.load(
                    f"local_data/records/experiments/{task.target_column}(MLP - Clinical data and manual radiomics)/"
                    f"outer_splits/split_{i}/best_model/best_model.pt")
            )

            model.fix_thresholds_to_optimal_values(dataset)
            model.fit_breslow_estimators(dataset)

            evaluator = ModelEvaluator(model=model, dataset=dataset)
            score = evaluator.compute_score(dataset.test_mask)

            c_index.append(score[task.name]["ConcordanceIndexCensored"])
            c_index_ipcw.append(score[task.name]["ConcordanceIndexIPCW"])
            cda.append(score[task.name]["CumulativeDynamicAUC"])

        print(task.target_column)
        print(f"Concordance index: {np.nanmean(c_index)} +/- {np.nanstd(c_index)}")
        print(f"Concordance index IPCW: {np.nanmean(c_index_ipcw)} +/- {np.nanstd(c_index_ipcw)}")
        print(f"Cumulative dynamic AUC: {np.nanmean(cda)} +/- {np.nanstd(cda)}")
