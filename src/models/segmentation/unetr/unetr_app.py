"""
    @file:              unetr_app.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains an implementation of a 3D UNETR.

"""

from src.models.segmentation.hdf_dataset import HDFDataset
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
import torch
from monai.transforms import (
    AddChannel,
    ToTensor,
    HistogramNormalize,
    CenterSpatialCrop,
    ThresholdIntensity,
    KeepLargestConnectedComponent,
    Compose
)
from torch.utils.data import random_split
from monai.networks.nets import UNETR
from monai.data.dataloader import DataLoader

if __name__ == '__main__':
    set_determinism(seed=1010710)

    writer = SummaryWriter(
        log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unetr/runs/exp1'
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
        ToTensor(dtype=torch.float32)
    ])
    seg_trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 160, 160)),
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

    ).to(device)

