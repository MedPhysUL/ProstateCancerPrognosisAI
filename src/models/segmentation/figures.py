import numpy as np
from monai.utils import set_determinism
from monai.data import DataLoader
from src.data.extraction.local import LocalDatabaseManager
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.visualization.image_viewer import ImageViewer
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    HistogramNormalized,
    KeepLargestConnectedComponentd,
    ThresholdIntensityd,
    ToTensord,
    RandZoomd,
    RandFlipd,
    RandShiftIntensityd,
    Rotate90d,
    RandGaussianNoised
)
import torch
from monai.utils import first
from torch.utils.data.dataset import random_split
from copy import deepcopy
import matplotlib.pyplot as plt
import h5py


if __name__ == '__main__':
    set_determinism(seed=1010710)

    file = h5py.File('C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5')

    z = 85

    # Single CT
    plt.imshow(file['TEP-005']['0']['Image'][:, :, z], cmap='gray')
    plt.show()

    # Single SEG
    plt.imshow(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z], cmap='gray')
    plt.show()

    # Single TEP
    plt.imshow(file['TEP-005']['1']['Image'][:, :, z], cmap='gray')
    plt.show()

    # CT + SEG
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(file['TEP-005']['0']['Image'][:, :, z], cmap='gray')
    axes[1].imshow(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z], alpha=1, cmap='gray')
    plt.show()

    # CT + SEG + TEP
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(file['TEP-005']['0']['Image'][:, :, z], cmap='gray')
    axes[1].imshow(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z], alpha=1, cmap='gray')
    axes[2].imshow(file['TEP-005']['1']['Image'][:, :, z], cmap='gray')
    plt.show()

    # CT w/ SEG
    plt.imshow(file['TEP-005']['0']['Image'][:, :, z], cmap='gray')
    plt.imshow(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z], alpha=0.1)
    plt.show()

    # TEP w/ SEG
    plt.imshow(file['TEP-005']['1']['Image'][:, :, z], cmap='gray')
    plt.imshow(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z], alpha=0.1)
    plt.show()

    # Shapes
    print('CT', np.shape(file['TEP-005']['0']['Image'][:, :, z]))
    print('TEP', np.shape(file['TEP-005']['1']['Image'][:, :, z]))
    print('SEG', np.shape(file['TEP-005']['0']['0']['Prostate_label_map'][:, :, z]))

    print('avec notre dataset')

    # Defining transforms
    trans = Compose([
        AddChanneld(keys=['img', 'tep',  'seg']),
        CenterSpatialCropd(keys=['img', 'tep', 'seg'], roi_size=(1000, 160, 160)),
        ThresholdIntensityd(keys=['img'], threshold=-250, above=True, cval=-250),
        ThresholdIntensityd(keys=['img'], threshold=500, above=False, cval=500),
        HistogramNormalized(keys=['img'], num_bins=751, min=0, max=1),
        ToTensord(keys=['img', 'tep', 'seg'], dtype=torch.float32)
    ])

    # ImageDataset
    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5'
        ),
        transform=trans,
        include_tep=True
    )

    # Dataset
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )

    z2 = 50

    print('IMG')

    plt.imshow(ds.image_dataset[1]['img'][0][z2], cmap='gray')
    plt.show()

    plt.imshow(ds.image_dataset[1]['seg'][0][z2], cmap='gray')
    plt.show()

    plt.imshow(ds.image_dataset[1]['tep'][0][z2], cmap='gray')
    plt.show()


