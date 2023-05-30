"""
    @file:              training_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 03/2023

    @Description:       This file contains the training of a variational inference (VI) prostate net.
"""

from delia.databases import PatientsDatabase
from monai.data import DataLoader
from monai.metrics import DiceMetric as MonaiDiceMetric
from monai.utils import set_determinism
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.losses.single_task import DiceLoss
from src.models.torch.segmentation.net.vi_prostate_net.vi_prostate_net import VIProstateNet
from src.tasks import SegmentationTask


if __name__ == "__main__":
    set_determinism(seed=1010710)

    writer = SummaryWriter(
        log_dir="../saved_parameters"
    )

    # Parameters for Training (part 1 of 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = 0
    num_val = 40
    batch_size = 8
    num_epochs = 250

    # Task
    task = SegmentationTask(
        criterion=DiceLoss(),
        organ="Prostate",
        modality="CT"
    )

    # Database
    database = PatientsDatabase(
        path_to_database=r"C:\Users\MALAR507\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\learning_set.h5"
    )

    # Dataset
    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=task
    )
    ds = ProstateCancerDataset(image_dataset=image_dataset)

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])

    # Data Loaders
    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn=None
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=None
    )

    # Model
    net = VIProstateNet(
        channels=(64, 128, 256, 512, 1024)      # TODO -- number of channels ?
    ).to(device)

    # Parameters for Training (part 2 of 2)     # TODO -- lr, temperature, dropout, weight decay, lr scheduler ?
    temperature = 1e-8
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)    # weight_decay=1e-3
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.99)
    # loss = nn.BCEWithLogitsLoss()
    from monai.losses import DiceLoss as MonaiDiceLoss
    loss = MonaiDiceLoss(sigmoid=True)
    metric = MonaiDiceMetric(include_background=True, reduction="mean")

    # Training Loop
    best_metric_avg = 0
    best_metric_avg_plus_min = 0

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_metrics_avg = []
    epoch_val_metrics_avg_plus_min = []

    for epoch in range(num_epochs):

        # Training
        net.train()

        batch_loss = []
        for batch in train_loader:
            batch_images = batch.x.image["CT"].to(device)
            batch_segs = batch.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"].to(device)

            opt.zero_grad()

            # Prediction
            y_pred, kl = net(batch_images)

            # Loss
            loss_train = loss(y_pred, batch_segs) + temperature * kl

            loss_train.backward()
            opt.step()

            batch_loss.append(loss_train.item())

        epoch_train_losses.append(np.average(batch_loss))
        writer.add_scalar("Average batch training vi loss per epoch", epoch_train_losses[-1], epoch + 1)

        # lr_scheduler.step()

        # Validation
        net.eval()

        loss_val_list = []
        metric_vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch_images = batch.x.image["CT"].to(device)
                batch_segs = batch.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"].to(device)

                # Prediction
                y_pred, kl = net(batch_images)

                # Loss
                loss_val = loss(y_pred, batch_segs) + temperature * kl
                loss_val_list.append(loss_val.item())

                # Post-processing
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)

                # Metric
                pred_metric = metric(y_pred=y_pred, y=batch_segs)
                metric_vals += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

        epoch_val_losses.append(np.average(loss_val_list))
        epoch_val_metrics_avg.append(np.average(metric_vals))
        epoch_val_metrics_avg_plus_min.append(np.average(metric_vals) + np.min(metric_vals))

        print(f"EPOCH {epoch + 1}, val metric: {epoch_val_metrics_avg[-1]}, val metric + worst: {epoch_val_metrics_avg_plus_min[-1]}")

        # Save Best Parameters
        if epoch_val_metrics_avg[-1] > best_metric_avg:
            best_metric_avg = epoch_val_metrics_avg[-1]
            torch.save(net.state_dict(), "../saved_parameters/best_parameters_avg.pt")
            print(f"New best metric avg: {best_metric_avg}")

        if epoch_val_metrics_avg_plus_min[-1] > best_metric_avg_plus_min:
            best_metric_avg_plus_min = epoch_val_metrics_avg_plus_min[-1]
            torch.save(net.state_dict(), "../saved_parameters/best_parameters_avg_plus_min.pt")
            print(f"New best metric avg+min: {best_metric_avg_plus_min}")

        writer.add_scalar("Average validation loss per epoch", epoch_val_losses[-1], epoch + 1)
        writer.add_scalar("Average validation metric per epoch", epoch_val_metrics_avg[-1], epoch + 1)
        writer.add_scalar("Average + min validation metric per epoc", epoch_val_metrics_avg_plus_min[-1], epoch + 1)

    writer.flush()
    writer.close()
