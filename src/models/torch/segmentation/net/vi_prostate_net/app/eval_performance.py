"""
    @file:              eval_performance.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This file contains a script to assess the performance of VIProstateNet.
"""

from delia.databases import PatientsDatabase
from monai.data import DataLoader
from monai.metrics import DiceMetric as MonaiDiceMetric
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split

from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.losses.single_task import DiceLoss
from src.models.torch.segmentation.net.vi_prostate_net.vi_prostate_net import VIProstateNet
from src.tasks import SegmentationTask
from src.visualization.image_viewer import ImageViewer


# What to show
best_parameters: str = "best_parameters_avg.pt"  # best_parameters_avg.pt or best_parameters_avg_plus_min.pt
show_all: bool = True
show_best_mid_worst: bool = True

if __name__ == "__main__":
    set_determinism(seed=1010710)

    # Parameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = MonaiDiceMetric(include_background=True, reduction="mean")
    num_val = 40
    num_workers = 0

    # Task
    task = SegmentationTask(
        criterion=DiceLoss(),
        organ="Prostate",
        modality="CT"
    )

    # Database
    database = PatientsDatabase(
        path_to_database=r"C:\Users\MALAR507\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\learning_set.h5"
    )

    # Dataset
    image_dataset = ImageDataset(
        database=database,
        modalities={"CT"},
        tasks=task
    )
    ds = ProstateCancerDataset(image_dataset=image_dataset)

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
    net = VIProstateNet(
        channels=(64, 128, 256, 512, 1024)
    ).to(device)

    # Load Best Parameters
    net.load_state_dict(torch.load(rf"C:\Users\MALAR507\Documents\GitHub\ProstateCancerPrognosisAI\src\models\torch\segmentation\net\vi_prostate_net\saved_parameters\{best_parameters}"))
    net.eval()

    # Show Stats
    metric_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch_images = batch.x.image["CT"]
            batch_segs = batch.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

            batch_images = batch_images.to(device)
            batch_segs = batch_segs.to(device)

            y_pred, _ = net(batch_images)      # Prediction

            y_pred = torch.sigmoid(y_pred)  # Post-processing
            y_pred = torch.round(y_pred)

            pred_metric = metric(y_pred=y_pred, y=batch_segs)
            metric_list += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

    print("Validation images metrics are:", metric_list)
    print("Max:", metric_list[np.argmax(metric_list)], "at index:", np.argmax(metric_list))
    print("Min:", metric_list[np.argmin(metric_list)], "at index:", np.argmin(metric_list))
    print("Median:", metric_list[np.argsort(metric_list)[len(metric_list) // 2]], "at index:",
          np.argsort(metric_list)[len(metric_list) // 2])
    print("Median score:", np.median(metric_list))
    print("Mean score:", np.average(metric_list))

    # Show Best-Mid-Worst
    if show_best_mid_worst:
        patient_idx = -1
        with torch.no_grad():
            for patient in val_loader:
                patient_img = patient.x.image["CT"]
                patient_seg = patient.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

                patient_idx += 1

                if patient_idx == np.argmax(metric_list):
                    print("Best image", patient_idx)

                    img = np.array(patient_img[0][0])
                    seg_truth = np.array(patient_seg[0][0])

                    patient_img = patient_img.to(device)

                    y_pred, _ = net(patient_img)  # Prediction

                    y_pred = torch.sigmoid(y_pred)  # Post-processing
                    y_pred = torch.round(y_pred)

                    print("Dice score", metric(y_pred=y_pred.cpu(), y=patient_seg))

                    seg_pred = np.array(y_pred[0][0].cpu())
                    ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=0.1)

                if patient_idx == np.argmin(metric_list):
                    print("Worst image:", patient_idx)

                    img = np.array(patient_img[0][0])
                    seg_truth = np.array(patient_seg[0][0])

                    patient_img = patient_img.to(device)

                    y_pred, _ = net(patient_img)  # Prediction

                    y_pred = torch.sigmoid(y_pred)  # Post-processing
                    y_pred = torch.round(y_pred)

                    print("Dice score", metric(y_pred=y_pred.cpu(), y=patient_seg))

                    seg_pred = np.array(y_pred[0][0].cpu())
                    ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=0.1)

                if patient_idx == np.argsort(metric_list)[len(metric_list) // 2]:
                    print("Median image", patient_idx)

                    img = np.array(patient_img[0][0])
                    seg_truth = np.array(patient_seg[0][0])

                    patient_img = patient_img.to(device)

                    y_pred, _ = net(patient_img)  # Prediction

                    y_pred = torch.sigmoid(y_pred)  # Post-processing
                    y_pred = torch.round(y_pred)

                    print("Dice score:", metric(y_pred=y_pred.cpu(), y=patient_seg))

                    seg_pred = np.array(y_pred[0][0].cpu())
                    ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=0.1)

    # Show All
    if show_all:
        with torch.no_grad():
            for patient in val_loader:
                patient_img = patient.x.image["CT"]
                patient_seg = patient.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

                img = np.array(patient_img[0][0])
                seg_truth = np.array(patient_seg[0][0])

                patient_img = patient_img.to(device)

                y_pred, _ = net(patient_img)  # Prediction

                y_pred = torch.sigmoid(y_pred)  # Post-processing
                y_pred = torch.round(y_pred)

                print("Dice score:", metric(y_pred=y_pred.cpu(), y=patient_seg))

                seg_pred = np.array(y_pred[0][0].cpu())
                ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)
