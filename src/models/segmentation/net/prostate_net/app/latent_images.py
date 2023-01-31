"""
    @file:              prostate_net.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       Description
"""

from monai.transforms import Compose, ToTensord, EnsureChannelFirstd
import torch
from torch.utils.data.dataset import random_split
from monai.utils import set_determinism
from src.data.extraction.local import LocalDatabaseManager
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.prostate_cancer_dataset import ProstateCancerDataset
from src.models.segmentation.net.prostate_net.prostate_net import ProstateNet
from src.utils.tasks import SegmentationTask
from src.utils.losses import DICELoss
from src.utils.score_metrics import DICEMetric
from monai.data import DataLoader
import numpy as np
from src.visualization.image_viewer import ImageViewer
import matplotlib.pyplot as plt


class LatentImages(ProstateNet):
    """
    Description.
    """
    def get_latent_images(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottom = self.bottom(enc4)
        dec4 = self.dec4(torch.cat([enc4, bottom], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

        return enc1, enc2, enc3, enc4, bottom, dec4, dec3, dec2, dec1


if __name__ == '__main__':
    set_determinism(seed=1010710)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load Net
    net = LatentImages(
        channels=(32, 64, 128, 256, 512),
    ).to(device)

    net.load_state_dict(torch.load(     # TODO
        'C:/Users/rapha/Documents/GitHub/ProstateCancerPrognosisAI/src/models/segmentation/net/prostate_net/saved_parameters/best_parameters_avg.pt'))

    net.eval()

    # Load Image
    transformations = Compose([
        EnsureChannelFirstd(keys=['CT', 'Prostate_segmentation']),
        ToTensord(keys=['CT', 'Prostate_segmentation'], dtype=torch.float32)
    ])
    task = SegmentationTask(
        criterion=DICELoss(),
        optimization_metric=DICEMetric(),
        organ="Prostate",
        modality="CT",
        evaluation_metrics=[DICEMetric()]
    )
    image_dataset = ImageDataset(
        database_manager=LocalDatabaseManager(
            path_to_database='C:/Users/rapha/Desktop/dummy_db.h5'   # TODO
        ),
        tasks=[task],
        modalities={"CT"},
        transforms=transformations
    )
    ds = ProstateCancerDataset(
        image_dataset=image_dataset
    )
    _, ds = random_split(ds, [len(ds) - 1, 1])
    loader = DataLoader(
        dataset=ds,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=None
    )

    # Get images
    with torch.no_grad():
        for patient in loader:
            img = patient.x.image['CT'].to(device)

            enc1, enc2, enc3, enc4, bottom, dec4, dec3, dec2, dec1 = net.get_latent_images(img)

            out = torch.sigmoid(dec1)
            out = torch.round(out).cpu()

    # Whole model view
    fig, ax = plt.subplot_mosaic("""
    A.........K
    .B.......J.
    ..C.....I..
    ...D...H...
    ....E.G....
    .....F.....
    """)
    ax["A"].imshow(img[0][0][80, :, :], cmap="Greys_r")
    ax["A"].set_title(f"img {np.shape(img)}")

    ax["B"].imshow(enc1[0][0][40, :, :], cmap="Greys_r")
    ax["B"].set_title(f"enc1 {np.shape(enc1)}")

    ax["C"].imshow(enc2[0][0][20, :, :], cmap="Greys_r")
    ax["C"].set_title(f"enc2 {np.shape(enc2)}")

    ax["D"].imshow(enc3[0][0][10, :, :], cmap="Greys_r")
    ax["D"].set_title(f"enc3 {np.shape(enc3)}")

    ax["E"].imshow(enc4[0][0][5, :, :], cmap="Greys_r")
    ax["E"].set_title(f"enc4 {np.shape(enc4)}")

    ax["F"].imshow(bottom[0][0][5, :, :], cmap="Greys_r")
    ax["F"].set_title(f"bottom {np.shape(bottom)}")

    ax["G"].imshow(dec4[0][0][10, :, :], cmap="Greys_r")
    ax["G"].set_title(f"dec4 {np.shape(dec4)}")

    ax["H"].imshow(dec3[0][0][20, :, :], cmap="Greys_r")
    ax["H"].set_title(f"dec3 {np.shape(dec3)}")

    ax["I"].imshow(dec2[0][0][40, :, :], cmap="Greys_r")
    ax["I"].set_title(f"dec2 {np.shape(dec2)}")

    ax["J"].imshow(dec1[0][0][80, :, :], cmap="Greys_r")
    ax["J"].set_title(f"dec1 {np.shape(dec1)}")

    ax["K"].imshow(out[0][0][80, :, :], cmap="Greys_r")
    ax["K"].set_title(f"out {np.shape(out)}")

    plt.show()

    # View a latent image
    ImageViewer().view_latent_image(dec3[0])        # layer[patient in batch], layer in img, enc1, ..., enc4, bottom, dec4, ..., dec1, out

    # View weights
    net_children = list(net.children())             # net_children[i], i from (0 = enc1, 1 = enc2, 2 = enc3, ...)
    ImageViewer().view_filter(net_children[2].conv1.weight)
