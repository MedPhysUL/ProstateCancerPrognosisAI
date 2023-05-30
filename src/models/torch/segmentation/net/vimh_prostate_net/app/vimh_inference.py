"""
    @file:              vimh_inference.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       Description.
"""

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import CenterSpatialCropd, Compose, EnsureChannelFirstd, ToTensord
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split

from src.data.extraction.local import LocalDatabaseManager
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.models.segmentation.net.vimh_prostate_net.vimh_prostate_net import VIMHProstateNet
from src.utils.losses import DICELoss
from src.utils.score_metrics import DICEMetric
from src.utils.tasks import SegmentationTask
from src.visualization.image_viewer import ImageViewer


best_parameters: str = "best_parameters_avg.pt"  # best_parameters_avg.pt or best_parameters_avg_plus_min.pt
patient_to_show: int = 1

if __name__ == "__main__":
    set_determinism(seed=1010710)

    # Parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = DiceMetric(include_background=True, reduction="mean")
    num_val = 2
    num_workers = 0

    # Transformations
    transformations = Compose([
        EnsureChannelFirstd(keys=["CT", "Prostate_segmentation"]),
        CenterSpatialCropd(keys=["CT", "Prostate_segmentation"], roi_size=(1000, 160, 160)),  # TODO
        ToTensord(keys=["CT", "Prostate_segmentation"], dtype=torch.float32)
    ])

    # Task
    task = SegmentationTask(
        criterion=DICELoss(),
        optimization_metric=DICEMetric(),
        organ="Prostate",
        modality="CT",
        evaluation_metrics=[DICEMetric()]
    )

    # Dataset
    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database="C:/Users/rapha/Desktop/dummy_db.h5"
        ),
        tasks=[task],
        modalities={"CT"},
        transforms=transformations
    )
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])

    # Dataloader
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=None
    )

    # Model
    net = VIMHProstateNet(
        num_heads=4,
        channels=(4, 8, 16, 32, 64)
    ).to(device)

    # Load Best Parameters
    net.load_state_dict(torch.load(
        f"C:/Users/rapha/Documents/GitHub/ProstateCancerPrognosisAI/src/models/segmentation/net/vimh_prostate_net/saved_parameters/{best_parameters}"))

    net.eval()

    with torch.no_grad():
        patient_idx = -1
        for patient in val_loader:
            patient_img = patient.x.image["CT"]
            patient_seg = patient.y["Prostate_segmentation"]

            patient_idx += 1

            if patient_idx == patient_to_show:
                print("Showing patient:", patient_idx)

                img = np.array(patient_img[0][0])
                seg_truth = np.array(patient_seg[0][0])

                patient_img = patient_img.to(device)

                y_pred, _ = net(patient_img)  # Prediction

                y_pred = torch.sigmoid(y_pred)  # Sigmoid

                ensemble_y_pred = torch.mean(y_pred, dim=0)  # Ensemble Prediction (mean)

                ensemble_y_pred = torch.round(ensemble_y_pred)  # Rounding

                seg_pred = np.array(ensemble_y_pred[0][0].cpu())
                ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=0.1)

                # Mean
                mean = np.mean(y_pred, axis=0)

                # Variance
                variance = np.var(y_pred, axis=0)


                print(torch.nn.BCELoss()(y_pred[2], patient_seg))
                print(np.max(mean))
                print(np.max(variance))




