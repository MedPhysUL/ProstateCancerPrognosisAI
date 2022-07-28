"""
    @file:              performances.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       This file contains a script to assess the performance of a 3D U-Net.

"""
import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AddChannel,
    CenterSpatialCrop,
    Compose,
    HistogramNormalize,
    KeepLargestConnectedComponent,
    ThresholdIntensity,
    ToTensor
)
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split

from src.data.extraction.local import LocalDatabaseManager
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.models.segmentation.hdf_dataset import HDFDataset
from src.visualization.image_viewer import ImageViewer

if __name__ == '__main__':
    set_determinism(seed=1010710)

    # Setting Up
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metric = DiceMetric(include_background=True, reduction='mean')
    num_val = 40
    num_workers = 0

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

    # Dataset
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )

    # Train/Val Split
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])

    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    # Model
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        dropout=0.2
    ).to(device)

    # Load Best Parameters
    net.load_state_dict(torch.load('C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete/best_model_parameters.pt'))
    net.eval()

    # Stats
    metric_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch_images = batch.image[0]
            batch_segs = batch.image[1]

            batch_images = batch_images.to(device)
            batch_segs = batch_segs.to(device)

            y_pred = net(batch_images)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)

            pred_metric = metric(y_pred=y_pred, y=batch_segs)
            metric_list += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

    print('les metriques des images de validation:', metric_list)
    print('max:', metric_list[np.argmax(metric_list)], 'at index:', np.argmax(metric_list))
    print('min:', metric_list[np.argmin(metric_list)], 'at index:', np.argmin(metric_list))
    print('mediane:', metric_list[np.argsort(metric_list)[len(metric_list)//2]], 'at:', np.argsort(metric_list)[len(metric_list)//2])
    print('np.mediane', np.median(metric_list))
    print('moyenne:', np.average(metric_list))

    # Show best-mid-worst
    patient_idx = -1
    with torch.no_grad():
        for patient in val_loader:
            patient_img = patient.image[0]
            patient_seg = patient.image[1]
            patient_idx += 1
            print(patient_idx)

            if patient_idx == np.argmax(metric_list):
                print('Best image', patient_idx)
                img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
                seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
                patient_img = patient_img.to(device)
                y_pred = net(patient_img)
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)
                print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
                seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
                ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=1)

            if patient_idx == np.argmin(metric_list):
                print('Worst image', patient_idx)
                img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
                seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
                patient_img = patient_img.to(device)
                y_pred = net(patient_img)
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)
                print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
                seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
                ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=1)

            if patient_idx == np.argsort(metric_list)[len(metric_list)//2]:
                print('Median image', patient_idx)
                img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
                seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
                patient_img = patient_img.to(device)
                y_pred = net(patient_img)
                y_pred = torch.sigmoid(y_pred)
                y_pred = torch.round(y_pred)
                print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
                seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
                ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred, alpha=1)

    # Show All
    with torch.no_grad():
        for patient in val_loader:
            patient_img = patient.image[0]
            patient_seg = patient.image[1]
            img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
            seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
            patient_img = patient_img.to(device)
            y_pred = net(patient_img)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)
            print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
            seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
            ImageViewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)

    # Tensorboard Model Graph
    from monai.utils import first
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/unet3d/runs/exp_delete')
    with torch.no_grad():
        img, seg = first(val_loader)
        img = img.to(device)
        writer.add_graph(net, img)
    writer.flush()
    writer.close()

    # Volume-Dice Plot
    metric_list = []
    volume_list = []
    with torch.no_grad():
        for batch in val_loader:
            batch_images = batch.image[0]
            batch_segs = batch.image[1]

            batch_images = batch_images.to(device)
            batch_segs = batch_segs.to(device)

            y_pred = net(batch_images)
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.round(y_pred)

            pred_metric = metric(y_pred=y_pred, y=batch_segs)
            metric_list += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]

            volume_list += [np.count_nonzero(batch_segs.cpu())]
    print(metric_list)
    print(volume_list)
    plt.scatter(x=volume_list, y=metric_list)
    plt.show()
