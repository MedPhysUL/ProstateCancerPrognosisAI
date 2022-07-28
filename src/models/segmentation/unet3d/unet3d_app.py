"""
    @file:              unet3d_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 07/2022

    @Description:       This file contains an implementation of a 3D U-Net.
"""

from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel,
    CenterSpatialCrop,
    Compose,
    HistogramNormalize,
    KeepLargestConnectedComponent,
    # RandFlip,
    # Rotate90,
    ThresholdIntensity,
    ToTensor
)
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

# from src.data.processing.copy_items import Augmentation, AugmentationTransforms
from src.data.extraction.local import LocalDatabaseManager
from src.data.processing.image_dataset import ImageDataset
from src.data.processing.prostate_cancer_dataset import ProstateCancerDataset
from src.models.segmentation.hdf_dataset import HDFDataset


if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter(
        log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete'
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_workers = 0
    num_val = 40
    batch_size = 4
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

    # ImageDataset
    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5'
        ),
        img_transform=img_trans,
        seg_transform=seg_trans
    )

    print("1")
    # Dataset
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )
    print("2")
    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])
    print("3")
    print(len(train_ds))
    ################
    # SHAPEEEEE
    # print(np.shape(train_ds))       # (230, 2)  faque (patients, task ou img/seg)
    # print(np.shape(train_ds[0]))    # (2,) task ou img/seg
    # print(np.shape(train_ds[0][0]))     # (2,) task ou img/seg
    # print(np.shape(train_ds[0][0][0]))  # torch.Size([1, 160, 160, 160]) (added, z, x, y)
    ########################################
    # Data Augmentation
    # aug_trans = [
    #     # Rotation 180 deg
    #     AugmentationTransforms(
    #         img_transforms=Compose([
    #             AddChannel(),
    #             Rotate90(k=2, spatial_axes=(1, 2)),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             ThresholdIntensity(threshold=-250, above=True, cval=-250),
    #             ThresholdIntensity(threshold=500, above=False, cval=500),
    #             HistogramNormalize(num_bins=751, min=0, max=1),
    #             ToTensor(dtype=torch.float32)
    #         ]),
    #         seg_transforms=Compose([
    #             AddChannel(),
    #             Rotate90(k=2, spatial_axes=(1, 2)),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             KeepLargestConnectedComponent(),
    #             ToTensor(dtype=torch.float32)
    #         ])
    #     ),
    #     # Flip lr
    #     AugmentationTransforms(
    #         img_transforms=Compose([
    #             AddChannel(),
    #             RandFlip(prob=1, spatial_axis=2),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             ThresholdIntensity(threshold=-250, above=True, cval=-250),
    #             ThresholdIntensity(threshold=500, above=False, cval=500),
    #             HistogramNormalize(num_bins=751, min=0, max=1),
    #             ToTensor(dtype=torch.float32)
    #         ]),
    #         seg_transforms=Compose([
    #             AddChannel(),
    #             RandFlip(prob=1, spatial_axis=2),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             KeepLargestConnectedComponent(),
    #             ToTensor(dtype=torch.float32)
    #         ])
    #     ),
    #     # Flip lr + Rotation 180 deg
    #     AugmentationTransforms(
    #         img_transforms=Compose([
    #             AddChannel(),
    #             RandFlip(prob=1, spatial_axis=2),
    #             Rotate90(k=2, spatial_axes=(1, 2)),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             ThresholdIntensity(threshold=-250, above=True, cval=-250),
    #             ThresholdIntensity(threshold=500, above=False, cval=500),
    #             HistogramNormalize(num_bins=751, min=0, max=1),
    #             ToTensor(dtype=torch.float32)
    #         ]),
    #         seg_transforms=Compose([
    #             AddChannel(),
    #             RandFlip(prob=1, spatial_axis=2),
    #             Rotate90(k=2, spatial_axes=(1, 2)),
    #             CenterSpatialCrop(roi_size=(1000, 160, 160)),
    #             KeepLargestConnectedComponent(),
    #             ToTensor(dtype=torch.float32)
    #         ])
    #     )
    # ]
    #
    # augmentation = Augmentation(augmentation_transforms=aug_trans)
    # train_ds_augmented = augmentation.get_augmented_dataset(train_ds)

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
    print("5")
    ##################################
    # Shapeeeee
    print(len(val_loader))  # 40
    for i in val_loader:
        # print(np.shape(i))      # (2,)
        for j in i:
            #print(np.shape(j))
            # donne ca :
            # (2,)
            # torch.Size([1])
            # hein
            for k in j:
                print(np.shape(k))
                # donnce ca:
                # torch.Size([1, 1, 160, 160, 160])
                # torch.Size([1, 1, 160, 160, 160])
                # torch.Size([])
                # hein
    #############################

    # Model
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

    # Training Loop
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_metrics = []
    best_metric = 0
    print("6")
    for epoch in range(num_epochs):
        net.train()
        batch_loss = []

        # Training
        for batch in train_loader:
            batch_images = batch.image[0].to(device)
            batch_segs = batch.image[1].to(device)
            #print("Table dataset exists?", not all(batch.table.isnan()))
            #print("7")
            opt.zero_grad()
            y_pred = net(batch_images)
            loss_train = loss(y_pred, batch_segs)
            loss_train.backward()
            opt.step()

            batch_loss.append(loss_train.item())
            #print("8")
        epoch_train_losses.append(np.average(batch_loss))
        writer.add_scalar('avg training loss per batch per epoch', epoch_train_losses[-1], epoch + 1)

        lr_scheduler.step()

        net.eval()
        loss_val_list = []
        metric_vals = []

        # Validation
        with torch.no_grad():
            for batch in val_loader:
                batch_images = batch.image[0].to(device)
                batch_segs = batch.image[1].to(device)

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
            torch.save(net.state_dict(), 'C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete/best_model_parameters.pt')

        writer.add_scalar('avg validation loss per epoch', epoch_val_losses[-1], epoch + 1)
        writer.add_scalar('avg validation metric per epoch', epoch_val_metrics[-1], epoch + 1)

    writer.flush()
    writer.close()

'''
        with torch.no_grad():
            for batch_images, batch_segs in val_loader:
                batch_images = batch_images.to(device)
                batch_segs = batch_segs.to(device)
'''