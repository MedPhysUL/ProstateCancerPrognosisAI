import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from delia.databases import PatientsDatabase

from constants import *
from src.data.processing.sampling import extract_masks, Mask
from src.data.datasets import ImageDataset, ProstateCancerDataset, TableDataset
from src.models.torch.extraction import CNN
from src.losses.multi_task import MeanLoss
from src.training import Trainer
from src.training.callbacks import LearningAlgorithm, Checkpoint
from src.training.callbacks.learning_algorithm import MultiTaskLossEarlyStopper
from src.training.callbacks.learning_algorithm.regularizer import L2Regularizer


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(DATA_PATH, "midl2023_learning_table.csv"))

    feature_cols = [AGE, PSA, GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    target_cols = [PN, BCR, BCR_TIME]

    df = df[[ID] + feature_cols + target_cols]

    table_dataset = TableDataset(
        df=df,
        ids_col=ID,
        tasks=PN_TASK,
        cont_cols=[AGE, PSA],
        cat_cols=[GLEASON_GLOBAL, GLEASON_PRIMARY, GLEASON_SECONDARY, CLINICAL_STAGE]
    )

    database = PatientsDatabase(path_to_database=r"local_data/midl2023_learning_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"PT", "CT"},
        organs={"CT": {"Prostate"}}
    )

    # Creation of the dataset
    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=table_dataset)

    masks = extract_masks(os.path.join(MASKS_PATH, "midl2023_masks.json"), k=1, l=0)

    dataset.update_masks(
        train_mask=masks[0][Mask.TRAIN],
        valid_mask=masks[0][Mask.VALID],
        test_mask=[]
    )

    model = CNN(
        in_shape=(1, 96, 96, 96),
        n_features=5,
        channels=(4, 8, 16, 32, 64),
        kernel_size=3,
        dropout=0.5,
        device=torch.device("cuda"),
        seed=SEED
    ).build(dataset)

    optimizer = Adam(
        params=model.parameters(),
        lr=3e-5,
        weight_decay=0.1
    )

    learning_algorithm = LearningAlgorithm(
        criterion=MeanLoss(),
        optimizer=optimizer,
        lr_scheduler=ExponentialLR(optimizer=optimizer, gamma=0.99),
        early_stopper=MultiTaskLossEarlyStopper(patience=20),
        regularizer=L2Regularizer(model.named_parameters(), lambda_=0.02)
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
