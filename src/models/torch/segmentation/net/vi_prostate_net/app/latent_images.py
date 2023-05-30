"""
    @file:              latent_images.py
    @Author:            Raphael Brodeur

    @Creation Date:     03/2022
    @Last modification: 03/2023

    @Description:       This files is used to visualize how the image input is modified throughout the model.
"""

from delia.databases import PatientsDatabase
import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from typing import NamedTuple

from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.losses.single_task import DiceLoss
from src.models.torch.segmentation.net.vi_prostate_net.vi_prostate_net import VIProstateNet
from src.tasks import SegmentationTask
from src.visualization.image_viewer import ImageViewer


class Layers(NamedTuple):
    img: torch.Tensor
    enc1: torch.Tensor
    enc2: torch.Tensor
    enc3: torch.Tensor
    enc4: torch.Tensor
    bottom: torch.Tensor
    dec4: torch.Tensor
    dec3: torch.Tensor
    dec2: torch.Tensor
    dec1: torch.Tensor


class LatentImages(VIProstateNet):
    """
    A child class of VIProstateNet that adds a method used to get the latent images while VIProstateNet forwards.
    """

    def get_latent_images(self, x: torch.Tensor) -> Layers:
        """
        Returns the latent images.

        Parameters
        ----------
        x : torch.Tensor
            Input of the model (a medical image).

        Returns
        -------
        layers : Layers
            Output image at every layer of the model.
        """
        enc1, _ = self.enc1(x)
        enc2, _ = self.enc2(enc1)
        enc3, _ = self.enc3(enc2)
        enc4, _ = self.enc4(enc3)
        bottom, _ = self.bottom(enc4)
        dec4, _ = self.dec4(torch.cat([enc4, bottom], dim=1))
        dec3, _ = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2, _ = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1, _ = self.dec1(torch.cat([enc1, dec2], dim=1))

        layers = Layers(
            img=x,
            enc1=enc1,
            enc2=enc2,
            enc3=enc3,
            enc4=enc4,
            bottom=bottom,
            dec4=dec4,
            dec3=dec3,
            dec2=dec2,
            dec1=dec1
        )

        return layers


if __name__ == "__main__":
    set_determinism(seed=100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Net
    net = LatentImages(
        channels=(64, 128, 256, 512, 1024),
    ).to(device)

    net.load_state_dict(torch.load(
        r"C:\Users\MALAR507\Documents\GitHub\ProstateCancerPrognosisAI\src\models\torch\segmentation\net\vi_prostate_net\saved_parameters\best_parameters_avg.pt"))

    net.eval()

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

    _, ds = random_split(ds, [len(ds) - 1, 1])

    loader = DataLoader(
        dataset=ds,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        collate_fn=None
    )

    # Get Images
    with torch.no_grad():
        for patient in loader:
            img = patient.x.image["CT"].to(device)

            layers = net.get_latent_images(img)

            out = torch.sigmoid(layers.dec1)
            # out = torch.round(out).cpu()

    # Whole Model View
    fig, ax = plt.subplot_mosaic("""
    A.........K
    .B.......J.
    ..C.....I..
    ...D...H...
    ....E.G....
    .....F.....
    """)
    size = int(np.shape(layers.img)[2])

    ax["A"].imshow(layers.img.cpu()[0][0][int(size / 2), :, :], cmap="Greys_r")
    ax["A"].set_title(f"img {np.shape(layers.img)}")

    ax["B"].imshow(layers.enc1.cpu()[0][0][int(size / 4), :, :], cmap="Greys_r")
    ax["B"].set_title(f"enc1 {np.shape(layers.enc1)}")

    ax["C"].imshow(layers.enc2.cpu()[0][0][int(size / 8), :, :], cmap="Greys_r")
    ax["C"].set_title(f"enc2 {np.shape(layers.enc2)}")

    ax["D"].imshow(layers.enc3.cpu()[0][0][int(size / 16), :, :], cmap="Greys_r")
    ax["D"].set_title(f"enc3 {np.shape(layers.enc3)}")

    ax["E"].imshow(layers.enc4.cpu()[0][0][int(size / 32), :, :], cmap="Greys_r")
    ax["E"].set_title(f"enc4 {np.shape(layers.enc4)}")

    ax["F"].imshow(layers.bottom.cpu()[0][0][int(size / 32), :, :], cmap="Greys_r")
    ax["F"].set_title(f"bottom {np.shape(layers.bottom)}")

    ax["G"].imshow(layers.dec4.cpu()[0][0][int(size / 16), :, :], cmap="Greys_r")
    ax["G"].set_title(f"dec4 {np.shape(layers.dec4)}")

    ax["H"].imshow(layers.dec3.cpu()[0][0][int(size / 8), :, :], cmap="Greys_r")
    ax["H"].set_title(f"dec3 {np.shape(layers.dec3)}")

    ax["I"].imshow(layers.dec2.cpu()[0][0][int(size / 4), :, :], cmap="Greys_r")
    ax["I"].set_title(f"dec2 {np.shape(layers.dec2)}")

    ax["J"].imshow(layers.dec1.cpu()[0][0][int(size / 2), :, :], cmap="Greys_r")
    ax["J"].set_title(f"dec1 {np.shape(layers.dec1)}")

    ax["K"].imshow(out.cpu()[0][0][int(size / 2), :, :], cmap="Greys_r")
    ax["K"].set_title(f"out {np.shape(out)}")

    plt.show()

    # View a Single Latent Image
    ImageViewer().view_latent_image(layers.enc1.cpu()[0])
