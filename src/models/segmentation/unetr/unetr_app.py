"""
    @file:              unetr_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains an implementation of a 3D UNETR.
"""

from monai.data.dataloader import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
from monai.transforms import (
    AddChannel,
    CenterSpatialCrop,
    Compose,
    HistogramNormalize,
    KeepLargestConnectedComponent,
    ThresholdIntensity,
    ToTensor
)
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from src.models.segmentation.hdf_dataset import HDFDataset


if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter(
        log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unetr/runs/exp_delete'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    num_val = 40
    batch_size = 1
    num_epochs = 150
    lr = 1e-3

    # Defining Transforms
    img_trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 160, 160)),
        ThresholdIntensity(threshold=-250, above=True, cval=-250),
        ThresholdIntensity(threshold=500, above=False, cval=500),
        HistogramNormalize(num_bins=751, min=0, max=1),
        ToTensor(dtype=torch.float32)
    ])
    seg_trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 160, 160)),
        KeepLargestConnectedComponent(),
        ToTensor(dtype=torch.float32)
    ])

    # Dataset
    ds = HDFDataset(
        path='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5',
        img_transform=img_trans,
        seg_transform=seg_trans
    )

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])

    # Data Loader
    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    # Model
    net = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(160, 160, 160),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed='conv',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr)
    loss = DiceLoss(sigmoid=True)
    metric = DiceMetric(include_background=True, reduction='mean')

    # Training Loop
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_metrics = []
    best_metric = 0
    for epoch in range(num_epochs):
        net.train()
        batch_loss = []

        # Training
        for batch in train_loader:
            batch_images = batch[0].to(device)
            batch_segs = batch[1].to(device)

            opt.zero_grad()
            y_pred = net(batch_images)

            loss_train = loss(y_pred, batch_segs)
            loss_train.backward()
            opt.step()

            batch_loss.append(loss_train.item())

        epoch_train_losses.append(np.average(batch_loss))
        writer.add_scalar('avg training loss per batch per epoch', epoch_train_losses[-1], epoch + 1)

        net.eval()
        loss_val_list = []
        metric_vals = []

        # Validation
        with torch.no_grad():
            for batch_images, batch_segs in val_loader:
                batch_images = batch_images.to(device)
                batch_segs = batch_segs.to(device)

                y_pred = net(batch_images)

                # Loss
                loss_val = loss(y_pred, batch_segs)
                loss_val_list.append(loss_val.item())

                # Post-processing
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)

                # Metric
                pred_metric = metric(y_pred=y_pred, y=batch_segs)
                metric_vals += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

        epoch_val_losses.append(np.average(loss_val_list))
        epoch_val_metrics.append(np.average(metric_vals))
        print(f"EPOCH {epoch + 1}, val metric : {epoch_val_metrics[-1]}")

        # Save Best Metric
        if epoch_val_metrics[-1] > best_metric:
            best_metric = epoch_val_metrics[-1]
            torch.save(net.state_dict(), 'C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unetr/runs/exp_delete/best_model_parameters.pt')

        writer.add_scalar('avg validation loss per epoch', epoch_val_losses[-1], epoch + 1)
        writer.add_scalar('avg validation metric per epoch', epoch_val_metrics[-1], epoch + 1)

    writer.flush()
    writer.close()
