import env_apps

from delia.databases import PatientsDatabase
from monai.transforms import (
    Compose,
    RandGaussianNoiseD,
    RandFlipD,
    RandRotateD,
    ThresholdIntensityD
)
import numpy as np
import pandas as pd
import torch

from constants import (
    ID,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    SEED,
    PROSTATE_SEGMENTATION_TASK,
    TABLE_TASKS, BCR_TASK, METASTASIS_TASK, HTX_TASK
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch import UNEXtractor


if __name__ == '__main__':
    for task in [BCR_TASK, METASTASIS_TASK, HTX_TASK]:
        df = pd.read_csv(LEARNING_TABLE_PATH)

        table_dataset = TableDataset(
            dataframe=df,
            ids_column=ID,
            tasks=task
        )

        database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

        image_dataset = ImageDataset(
            database=database,
            modalities={"PT", "CT"},
            tasks=PROSTATE_SEGMENTATION_TASK,
            augmentations=Compose([
                RandGaussianNoiseD(keys=["CT", "PT"], prob=0.5, std=0.05),
                ThresholdIntensityD(keys=["CT", "PT"], threshold=0, above=True, cval=0),
                ThresholdIntensityD(keys=["CT", "PT"], threshold=1, above=False, cval=1),
                RandFlipD(keys=["CT", "PT", "CT_Prostate"], prob=0.5, spatial_axis=2),
                RandRotateD(
                    keys=["CT", "PT", "CT_Prostate"],
                    mode=["bilinear", "bilinear", "nearest"],
                    prob=0.5,
                    range_x=0.174533
                )
            ]),
            seed=SEED
        )

        dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

        masks = extract_masks(MASKS_PATH, k=5, l=5)
        seg, c_index, c_index_ipcw, cda = [], [], [], []
        for i in range(5):
            dataset.update_masks(
                train_mask=masks[i][Mask.TRAIN],
                test_mask=masks[i][Mask.TEST],
                valid_mask=masks[i][Mask.VALID]
            )

            model = UNEXtractor(
                image_keys=["CT", "PT"],
                dropout_cnn=0.2,
                dropout_fnn=0.2,
                num_res_units=2,
                device=torch.device("cuda"),
                seed=SEED
            ).build(dataset)

            loaded_state = torch.load(
                f"local_data/records/experiments/{task.target_column}(UNEXtractor - Deep radiomics)/"
                f"outer_splits/split_{i}/best_model/best_model.pt"
            )

            model.load_state_dict(loaded_state)
            model.fix_thresholds_to_optimal_values(dataset)
            model.fit_breslow_estimators(dataset)

            score = model.compute_score_on_dataset(dataset, dataset.test_mask)

            c_index.append(score[task.name]["ConcordanceIndexCensored"])
            c_index_ipcw.append(score[task.name]["ConcordanceIndexIPCW"])
            cda.append(score[task.name]["CumulativeDynamicAUC"])
            seg.append(score[PROSTATE_SEGMENTATION_TASK.name]["DiceMetric"])

        print(task.target_column)
        print(f"Concordance index: {np.nanmean(c_index)} +/- {np.nanstd(c_index)}")
        print(f"Concordance index IPCW: {np.nanmean(c_index_ipcw)} +/- {np.nanstd(c_index_ipcw)}")
        print(f"Cumulative dynamic AUC: {np.nanmean(cda)} +/- {np.nanstd(cda)}")
        print(f"Segmentation: {np.nanmean(seg)} +/- {np.nanstd(seg)}")
