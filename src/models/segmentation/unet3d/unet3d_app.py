"""
    @file:              unet3d_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 06/2022

    @Description:       This file contains an implementation of a 3D U-Net.

"""
import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset

from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AddChannel, CenterSpatialCrop, Compose, ToTensor
from monai.utils import set_determinism
import torch
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    num_val = 30
    batch_size = 1
    num_epochs = 3
    lr = 1e-4

    trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 128, 128)),
        ToTensor(dtype=torch.float32)
    ])

    ds = HDFDataset(
        path='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5',
        img_transform=trans,
        seg_transform=trans
    )

    train_ds = ds[:-num_val]
    val_ds = ds[-num_val:]

    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 1, 1, 1)
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr)
    loss = DiceLoss(sigmoid=True)
    metric = DiceMetric(include_background=True, reduction='mean')

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_metrics = []

    for epoch in range(num_epochs):
        net.train()
        batch_loss = []

        for batch in train_loader:
            batch_images = batch[0].to(device)
            batch_segs = batch[1].to(device)

            opt.zero_grad()
            y_pred = net(batch_images)

            y_pred = torch.sigmoid(y_pred)

            loss_train = loss(y_pred, batch_segs)
            loss_train.backward()
            opt.step()

            batch_loss.append(loss_train.item())

        epoch_train_losses.append(np.average(batch_loss))
        writer.add_scalar('avg training loss per batch per epoch', epoch_train_losses[-1], epoch + 1)

        net.eval()
        loss_val_list = []
        metric_vals = []

        with torch.no_grad():
            for batch_images, batch_segs in val_loader:
                batch_images = batch_images.to(device)
                batch_segs = batch_segs.to(device)

                y_pred = net(batch_images)

                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)

                # Loss
                loss_val = loss(y_pred, batch_segs)
                loss_val_list.append(loss_val.item())

                # Metric
                pred_metric = metric(y_pred=y_pred, y=batch_segs)
                metric_vals += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

        epoch_val_losses.append(np.average(loss_val_list))
        epoch_val_metrics.append(np.average(metric_vals))
        print(f"EPOCH {epoch + 1}, val metric : {epoch_val_metrics[-1]}")
        writer.add_scalar('avg validation loss per epoch', epoch_val_losses[-1], epoch + 1)
        writer.add_scalar('avg validation metric per epoch', epoch_val_metrics[-1], epoch + 1)

    writer.flush()
    writer.close()
