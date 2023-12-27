import env_apps

import os

from delia.databases import PatientsDatabase
from monai.transforms import (
    Compose,
    RandGaussianNoiseD,
    RandFlipD,
    RandRotateD,
    ThresholdIntensityD
)
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from constants import (
    AUTOMATIC_BAYESIAN_RADIOMICS_MODELS_PATH,
    LEARNING_SET_PATH,
    MASKS_PATH,
    PROSTATE_SEGMENTATION_TASK,
    SEED
)
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.data.processing.sampling import extract_masks, Mask
from src.models.torch.segmentation import Unet
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=PROSTATE_SEGMENTATION_TASK,
        augmentations=Compose([
            RandGaussianNoiseD(keys=["CT"], prob=0.5, std=0.05),
            ThresholdIntensityD(keys=["CT"], threshold=0, above=True, cval=0),
            ThresholdIntensityD(keys=["CT"], threshold=1, above=False, cval=1),
            RandFlipD(keys=["CT", "CT_Prostate"], prob=0.5, spatial_axis=2),
            RandRotateD(
                keys=["CT", "CT_Prostate"],
                mode=["bilinear", "nearest"],
                prob=0.5,
                range_x=0.174533
            )
        ]),
        seed=SEED
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    masks = extract_masks(MASKS_PATH, k=5, l=5)

    for k, v in masks.items():

        dataset.update_masks(
            train_mask=v[Mask.TRAIN],
            test_mask=v[Mask.TEST],
            valid_mask=v[Mask.VALID]
        )

        model = Unet(
            image_keys="CT",
            num_res_units=3,
            bayesian=True,
            temperature={PROSTATE_SEGMENTATION_TASK.name: 1e-4},
            device=torch.device("cuda"),
            seed=SEED
        ).build(dataset)

        optimizer = Adam(
            params=model.parameters(),
            lr=1e-3
        )

        learning_algorithm = LearningAlgorithm(
            criterion=MeanLoss(),
            optimizer=optimizer,
            lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
            early_stopper=MultiTaskLossEarlyStopper(patience=20)
        )

        trainer = Trainer(
            batch_size=16,
            checkpoint=Checkpoint(
                path_to_checkpoint_folder=os.path.join(AUTOMATIC_BAYESIAN_RADIOMICS_MODELS_PATH, f"outer_split_{k}"),
                save_freq=20
            ),
            exec_metrics_on_train=True,
            n_epochs=150,
            seed=SEED
        )

        trained_model, history = trainer.train(
            model=model,
            dataset=dataset,
            learning_algorithms=learning_algorithm
        )

        # history.plot(show=True)
