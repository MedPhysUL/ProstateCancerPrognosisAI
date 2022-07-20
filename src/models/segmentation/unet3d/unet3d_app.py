"""
    @file:              unet3d_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains an implementation of a 3D U-Net.

"""
from copy import deepcopy

import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset

from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AddChannel, CenterSpatialCrop, Compose, KeepLargestConnectedComponent, ThresholdIntensity, \
    ToTensor, HistogramNormalize
from monai.utils import set_determinism
import torch
# from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter(log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    num_val = 29
    batch_size = 2
    num_epochs = 120
    lr = 1e-3

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

    ds = HDFDataset(
        path='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5',
        img_transform=img_trans,
        seg_transform=seg_trans
    )

    train_ds = ds[:-num_val]
    # print(type(train_ds))
    # print(len(train_ds.dataset.data[0].transform))
    # print(len(train_ds.dataset.data[1].transform))
    val_ds = ds[-num_val:]

    print(len(ds))
    # train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])
    print(len(train_ds))
    train_ds2 = deepcopy(train_ds)
    train_ds2.dataset.data[0].transform = Compose([AddChannel(), CenterSpatialCrop(roi_size=(1000, 160, 160)), ToTensor(dtype=torch.float32)])
    print(len(train_ds.dataset.data[0].transform))
    print(len(train_ds2.dataset.data[0].transform))

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
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        dropout=0.2
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.99)
    loss = DiceLoss(sigmoid=True)
    metric = DiceMetric(include_background=True, reduction='mean')

    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_metrics = []
    best_metric = 0

    for epoch in range(num_epochs):
        net.train()
        batch_loss = []

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

        lr_scheduler.step()

        net.eval()
        loss_val_list = []
        metric_vals = []

        with torch.no_grad():
            for batch_images, batch_segs in val_loader:
                batch_images = batch_images.to(device)
                batch_segs = batch_segs.to(device)

                y_pred = net(batch_images)

                # Loss
                loss_val = loss(y_pred, batch_segs)
                loss_val_list.append(loss_val.item())

                # Metric
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)

                pred_metric = metric(y_pred=y_pred, y=batch_segs)
                metric_vals += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

        epoch_val_losses.append(np.average(loss_val_list))
        epoch_val_metrics.append(np.average(metric_vals))
        print(f"EPOCH {epoch + 1}, val metric : {epoch_val_metrics[-1]}")

        if epoch_val_metrics[-1] > best_metric:
            best_metric = epoch_val_metrics[-1]
            torch.save(net.state_dict(), 'C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete/best_model_parameters.pt')

        writer.add_scalar('avg validation loss per epoch', epoch_val_losses[-1], epoch + 1)
        writer.add_scalar('avg validation metric per epoch', epoch_val_metrics[-1], epoch + 1)

    writer.flush()
    writer.close()
