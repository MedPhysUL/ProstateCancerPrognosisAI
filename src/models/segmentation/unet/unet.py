"""
    @file:              hdf_dataset.py
    @Author:            Raphael Brodeur

    @Creation Date:     05/2022
    @Last modification: 05/2022

    @Description:       This file contains an implementation of a U-Net.

"""

import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset

from monai.data import DataLoader
from monai.utils import set_determinism
import torch


# Set determinism
set_determinism(seed=1010710)

# Setting up
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_val = 1
batch_size = 4
num_workers = 1
num_epochs = 30
lr = 5e-3

# Setting up data

# img_trans
# seg_trans

ds = HDFDataset('C:/Users/rapha/Desktop/patients_dataset.h5')
# Visualise
print('ds shape:', np.shape(ds))
# (Patient, Channels, Z, X, Y)

train_ds = ds[:-num_val]
val_ds = ds[-num_val:]
# Visualise
print('train_ds shape:', np.shape(train_ds))
print('val_ds shape:', np.shape(val_ds))
# (Patient, Channels, Z, X, Y)

train_loader = DataLoader(
    dataset=train_ds,
    num_workers=num_workers,
    batch_size=batch_size
)
val_loader = DataLoader(
    dataset=val_ds,
    num_workers=num_workers,
    batch_size=batch_size
)
#Visualize
print('train_loader shape:', np.shape(train_loader))
print('train_loader shape:', np.shape(val_loader))











