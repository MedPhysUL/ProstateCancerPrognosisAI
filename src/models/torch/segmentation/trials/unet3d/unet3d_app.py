"""
    @file:              unet3d_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 09/2022

    @Description:       This file contains an implementation of a 3D U-Net.
"""

from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    HistogramNormalized,
    KeepLargestConnectedComponentd,
    ThresholdIntensityd,
    ToTensord, EnsureChannelFirstd,
)
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from src.data.extraction.local import LocalDatabaseManager
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.utils.tasks import SegmentationTask
from src.utils.losses import DICELoss
from src.utils.score_metrics import DICEMetric


if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter(
        log_dir='/applications/local_data/unet3d/runs/exp01'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    num_val = 1
    batch_size = 1
    num_epochs = 4
    lr = 1e-3

    # Defining Transforms
    trans = Compose([
        EnsureChannelFirstd(keys=['CT', 'Prostate_segmentation']),
        CenterSpatialCropd(keys=['CT', 'Prostate_segmentation'], roi_size=(1000, 160, 160)),
        # ThresholdIntensityd(keys=['img'], threshold=-250, above=True, cval=-250),
        # ThresholdIntensityd(keys=['img'], threshold=500, above=False, cval=500),
        # HistogramNormalized(keys=['img'], num_bins=751, min=0, max=1),
        # KeepLargestConnectedComponentd(keys=['seg']),
        ToTensord(keys=['CT', 'Prostate_segmentation'], dtype=torch.float32)
    ])

    # ImageDataset
    task = SegmentationTask(
        criterion=DICELoss(),
        optimization_metric=DICEMetric(),
        organ="Prostate",
        modality="CT",
        evaluation_metrics=[DICEMetric()]
    )

    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database='C:/Users/rapha/Desktop/dummy_db.h5'
        ),
        tasks=[task],
        modalities={"CT"},
        transforms=trans
    )

    # Dataset
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])


    # Data Loader
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
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(4, 8, 16, 32, 64),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        dropout=0.2
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.99)
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
            batch_images = batch.x.image['CT'].to(device)
            batch_segs = batch.y['Prostate_segmentation'].to(device)

            opt.zero_grad()
            y_pred = net(batch_images)
            loss_train = loss(y_pred, batch_segs)
            loss_train.backward()
            opt.step()

            batch_loss.append(loss_train.item())

        epoch_train_losses.append(np.average(batch_loss))
        writer.add_scalar('avg training loss per batch per epoch', epoch_train_losses[-1], epoch + 1)

        lr_scheduler.step()

        # Validation
        net.eval()
        loss_val_list = []
        metric_vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch_images = batch.x.image['CT'].to(device)
                batch_segs = batch.y["Prostate_segmentation"].to(device)

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
            torch.save(net.state_dict(), '/applications/local_data/unet3d/runs/exp01/best_model_parameters.pt')

        writer.add_scalar('avg validation loss per epoch', epoch_val_losses[-1], epoch + 1)
        writer.add_scalar('avg validation metric per epoch', epoch_val_metrics[-1], epoch + 1)

    writer.flush()
    writer.close()

    print(net)
    print("param", sum(param.numel() for param in net.parameters()))
    print("trainable", sum(p.numel() for p in net.parameters() if p.requires_grad))

    from torchsummary import summary
    summary(net)
    