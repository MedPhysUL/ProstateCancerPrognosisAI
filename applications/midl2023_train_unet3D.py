import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from delia.databases import PatientsDatabase
import numpy as np
from sklearn.model_selection import train_test_split

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.models.torch.segmentation import Unet
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=r"local_data/midl2023_learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    train_mask, valid_mask = train_test_split(np.arange(229), test_size=0.15, random_state=SEED)

    dataset.update_masks(
        train_mask=train_mask.tolist(),
        valid_mask=valid_mask.tolist(),
        test_mask=[]
    )

    model = Unet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        dropout=0.2,
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
        batch_size=8,
        checkpoint=Checkpoint(),
        exec_metrics_on_train=True,
        n_epochs=100,
        seed=SEED
    )

    trained_model, history = trainer.train(
        model=model,
        dataset=dataset,
        learning_algorithms=learning_algorithm
    )

    history.plot(show=True)
