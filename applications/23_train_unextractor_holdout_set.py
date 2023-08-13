import env_apps
import json
import os

from delia.databases import PatientsDatabase
from monai.transforms import (
    Compose,
    RandGaussianNoiseD,
    RandFlipD,
    RandRotateD,
    ThresholdIntensityD
)
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    BCR_TASK,
    EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT,
    ID,
    HOLDOUT_MASKS_PATH,
    LEARNING_TABLE_PATH,
    MASKS_PATH,
    PROSTATE_SEGMENTATION_TASK,
    SEED,
    LEARNING_SET_PATH
)
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch.extraction import UNEXtractor
from src.losses.multi_task import MeanLoss, WeightedSumLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    DOCKER_EXPERIMENTS_PATH = "experiments"
    TEMP_PATH = "temp"

    df = pd.read_csv(LEARNING_TABLE_PATH)

    table_dataset = TableDataset(
        dataframe=df,
        ids_column=ID,
        tasks=BCR_TASK
    )

    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

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

    masks = json.load(open(HOLDOUT_MASKS_PATH, "r"))
    dataset.update_masks(
        train_mask=masks[Mask.TRAIN],
        valid_mask=masks[Mask.VALID]
    )

    model = UNEXtractor(
        image_keys=["CT", "PT"],
        dropout_cnn=0.6,
        dropout_fnn=0.3,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=0.0004,
        weight_decay=0.003
    )

    learning_algorithm = LearningAlgorithm(
        criterion=WeightedSumLoss(weights=[1/2, 1/2], tasks=[BCR_TASK, PROSTATE_SEGMENTATION_TASK]),
        optimizer=optimizer,
        clip_grad_max_norm=EXTRACTOR_CLIP_GRAD_MAX_NORM_DICT[BCR_TASK.target_column],
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(criterion=MeanLoss(tasks=BCR_TASK), patience=30),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.005)
    )

    path_to_record_folder = os.path.join(
        os.getcwd(),
        DOCKER_EXPERIMENTS_PATH,
        f"UNEXtractor - Holdout set"
    )

    trainer = Trainer(
        batch_size=16,
        checkpoint=Checkpoint(path_to_checkpoint_folder=path_to_record_folder, save_freq=-1),
        exec_metrics_on_train=True,
        n_epochs=150,
        seed=SEED
    )

    path_to_temp_folder = os.path.join(
        os.getcwd(),
        TEMP_PATH,
        f"UNEXtractor - Holdout set"
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm,
        path_to_temporary_folder=path_to_temp_folder
    )

    # history.plot(show=True)
