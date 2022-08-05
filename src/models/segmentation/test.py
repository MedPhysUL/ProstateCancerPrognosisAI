import numpy as np
from monai.utils import set_determinism
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
    ToTensord
)
import torch


if __name__ == '__main__':
    set_determinism(seed=1010710)

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

    img = np.transpose(np.array(ds.image_dataset[0]['img'][0]), (1, 2, 0))
    seg = np.transpose(np.array(ds.image_dataset[0]['seg'][0]), (1, 2, 0))
    ImageViewer().compare(img=img, seg_truth=seg, seg_pred=seg)
