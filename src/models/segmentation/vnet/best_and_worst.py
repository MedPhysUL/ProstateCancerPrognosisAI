import numpy as np

from src.models.segmentation.hdf_dataset import HDFDataset
from src.visualization.viewer import Viewer

from monai.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AddChannel, CenterSpatialCrop, Compose, ToTensor, ThresholdIntensity, HistogramNormalize, \
    KeepLargestConnectedComponent, ScaleIntensityRange
from monai.networks.nets import UNet, VNet
from monai.utils import set_determinism
from torch.utils.data.dataset import random_split
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Setting Up Exp
    set_determinism(seed=1010710)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metric = DiceMetric(include_background=True, reduction='mean')
    num_val = 40
    num_workers = 0
    batch_size = 1

    # Defining Transforms
    img_trans = Compose([
        AddChannel(),
        CenterSpatialCrop(roi_size=(1000, 160, 160)),
        ScaleIntensityRange(a_min=-250, a_max=500, b_max=1.0, b_min=0.0, clip=True),
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
    train_ds, val_ds = random_split(ds, [len(ds) - num_val, num_val])

    val_loader = DataLoader(
        dataset=val_ds,
        num_workers=num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    # Model
    net = VNet(dropout_prob=0.9).to(device)

    # Load Best Parameters
    net.load_state_dict(torch.load('C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/vnet/runs/exp1/best_model_parameters.pt'))
    net.eval()

    # Stats
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
    print('np.mediane', np.median(metric_list))
    print('moyenne:', np.average(metric_list))

    # Show best-mid-worst
    # patient_idx = -1
    # with torch.no_grad():
    #     for patient_img, patient_seg in val_loader:
    #         patient_idx += 1
    #         print(patient_idx)
    #
    #         if patient_idx == np.argmax(metric_list):
    #             print('Best image', patient_idx)
    #             img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
    #             seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
    #             patient_img = patient_img.to(device)
    #             y_pred = net(patient_img)
    #             y_pred = torch.sigmoid(y_pred)
    #             y_pred = torch.round(y_pred)
    #             print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
    #             seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
    #             Viewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)
    #
    #         if patient_idx == np.argmin(metric_list):
    #             print('Worst image', patient_idx)
    #             img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
    #             seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
    #             patient_img = patient_img.to(device)
    #             y_pred = net(patient_img)
    #             y_pred = torch.sigmoid(y_pred)
    #             y_pred = torch.round(y_pred)
    #             print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
    #             seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
    #             Viewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)
    #
    #         if patient_idx == np.argsort(metric_list)[len(metric_list)//2]:
    #             print('Median image', patient_idx)
    #             img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
    #             seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
    #             patient_img = patient_img.to(device)
    #             y_pred = net(patient_img)
    #             y_pred = torch.sigmoid(y_pred)
    #             y_pred = torch.round(y_pred)
    #             print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
    #             seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
    #             Viewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)

    # Show All
    # with torch.no_grad():
    #     for patient_img, patient_seg in val_loader:
    #         img = np.transpose(np.array(patient_img[0][0]), (1, 2, 0))
    #         seg_truth = np.transpose(np.array(patient_seg[0][0]), (1, 2, 0))
    #         patient_img = patient_img.to(device)
    #         y_pred = net(patient_img)
    #         y_pred = torch.sigmoid(y_pred)
    #         y_pred = torch.round(y_pred)
    #         print('dice score:', metric(y_pred=y_pred.cpu(), y=patient_seg))
    #         seg_pred = np.transpose(np.array(y_pred[0][0].cpu()), (1, 2, 0))
    #         Viewer().compare(img=img, seg_truth=seg_truth, seg_pred=seg_pred)

    # Tensorboard model graph
    # from monai.utils import first
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter(log_dir='C:/Users/CHU/Documents/GitHub/ProstateCancerPrognosisAI/applications/local_data/vnet/runs/exp1')
    # with torch.no_grad():
    #     img, seg = first(val_loader)
    #     img = img.to(device)
    #     writer.add_graph(net, img)
    # writer.flush()
    # writer.close()

    # Volume-Dice Plot
    # metric_list = []
    # volume_list = []
    # with torch.no_grad():
    #     for batch_images, batch_segs in val_loader:
    #         batch_images = batch_images.to(device)
    #         batch_segs = batch_segs.to(device)
    #
    #         y_pred = net(batch_images)
    #         y_pred = torch.sigmoid(y_pred)
    #         y_pred = torch.round(y_pred)
    #
    #         pred_metric = metric(y_pred=y_pred, y=batch_segs)
    #         metric_list += [i for i in pred_metric.cpu().data.numpy().flatten().tolist()]
    #
    #         volume_list += [np.count_nonzero(batch_segs.cpu())]
    # print(metric_list)
    # print(volume_list)
    # plt.scatter(x=volume_list, y=metric_list)
    # plt.show()
