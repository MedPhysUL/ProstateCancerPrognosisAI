"""
    @file:              best_and_worst.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2022
    @Last modification: 07/2022

    @Description:       Description.

"""
import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset
from src.visualization.viewer import Viewer

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AddChannel, CenterSpatialCrop, Compose, ToTensor
from monai.networks.nets import UNet
from monai.utils import set_determinism
import torch


if __name__ == '__main__':
    set_determinism(seed=1010710)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metric = DiceMetric(include_background=True, reduction='mean')
    num_val = 30
    num_workers = 0
    batch_size = 1

    # Load validation set
    trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 128, 128)),
        ToTensor(dtype=torch.float32)
    ])
    ds = HDFDataset(
        path='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/learning_set.h5',
        img_transform=trans,
        seg_transform=trans
    )
    val_ds = ds[-num_val:]
    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    # Load model
    net = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(1, 1, 1, 1)
    ).to(device)
    net.load_state_dict(torch.load('C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/src/models/segmentation/unet3d/best_model_parameters.pt'))

    net.eval()
    metric_list = []

    with torch.no_grad():
        for batch_images, batch_segs in val_loader:
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
    print('moyenne:', np.average(metric_list))
