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
    RandShiftIntensityd
)
import torch
from monai.utils import first
from torch.utils.data.dataset import random_split
from copy import deepcopy

if __name__ == '__main__':
    set_determinism(seed=1010710)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Defining Transforms
    trans = Compose([
        AddChanneld(keys=['img', 'seg']),
        CenterSpatialCropd(keys=['img', 'seg'], roi_size=(1000, 160, 160)),
        ThresholdIntensityd(keys=['img'], threshold=-250, above=True, cval=-250),
        ThresholdIntensityd(keys=['img'], threshold=500, above=False, cval=500),
        HistogramNormalized(keys=['img'], num_bins=751, min=0, max=1),
        KeepLargestConnectedComponentd(keys=['seg']),
        ToTensord(keys=['img', 'seg'], dtype=torch.float32)
    ])

    # ImageDataset
    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5'
        ),
        transform=trans,
    )

    # Dataset
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - 10, 10])

    # Loader
    loader = DataLoader(
        dataset=val_ds,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    train_loader = DataLoader(
        dataset=train_ds,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    # print('A', np.shape(loader))    # A ()
    # print('B', np.shape(loader.dataset))    # B (270, 2)
    # print('C', np.shape(loader.dataset[0])) # C (2,)
    # print('D', np.shape(loader.dataset[0].image))   # D ()
    # print('E', np.shape(loader.dataset[0].image['img']))    # E torch.Size([1, 160, 160, 160])

    # loader
    # img = np.transpose(np.array(loader.dataset[0].image['img'][0]), (1, 2, 0))
    # seg_truth = np.transpose(np.array(loader.dataset[0].image['seg'][0]), (1, 2, 0))
    # seg_pred = np.transpose(np.array(loader.dataset[0].image['seg'][0]), (1, 2, 0))
    # ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=0.1)

    loader2 = deepcopy(loader)

    loader2.dataset.dataset.image_dataset.transform = Compose([
        AddChanneld(keys=['img', 'seg']),
        CenterSpatialCropd(keys=['img', 'seg'], roi_size=(1000, 160, 160)),
        RandZoomd(keys=['img', 'seg'], max_zoom=0.6, min_zoom=0.5, prob=0.5),
        ToTensord(keys=['img', 'seg'], dtype=torch.float32)
    ])
    print(len(loader2))
    # loader
    for batch in loader2:
        batch_images = batch.image['img']
        batch_segs = batch.image['seg']

        img = np.transpose(np.array(batch_images[0][0]), (1, 2, 0))
        seg_truth = np.transpose(np.array(batch_segs[0][0]), (1, 2, 0))
        ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_truth, alpha=0.1)

    print('salut')
    print(len(train_loader))
    for batch in train_loader:
        batch_images = batch.image['img']
        batch_segs = batch.image['seg']

        img = np.transpose(np.array(batch_images[0][0]), (1, 2, 0))
        seg_truth = np.transpose(np.array(batch_segs[0][0]), (1, 2, 0))
        ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_truth, alpha=0.1)


