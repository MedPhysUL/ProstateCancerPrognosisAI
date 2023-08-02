"""
    @file:              latent_images.py
    @Author:            Raphael Brodeur

    @Creation Date:     12/2022
    @Last modification: 01/2023

    @Description:       This files is used to visualize how the image input is modified throughout the model and to
                        visualize convolution kernels.
"""

from delia.databases import PatientsDatabase
import matplotlib.pyplot as plt
from monai.data import DataLoader
from monai.metrics import DiceMetric as MonaiDiceMetric
from monai.utils import set_determinism
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from typing import NamedTuple

from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.losses.single_task import DiceLoss
from src.models.torch.segmentation.net.prostate_net.prostate_net import ProstateNet
from src.tasks import SegmentationTask
from src.visualization.image_viewer import ImageViewer


# Parameters
show_layer_stats: bool = False
save_latent_images: bool = True


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


class LatentImages(ProstateNet):
    """
    A child class of ProstateNet that adds a method used to get the latent images while ProstateNet forwards.
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
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        bottom = self.bottom(enc4)
        dec4 = self.dec4(torch.cat([enc4, bottom], dim=1))
        dec3 = self.dec3(torch.cat([enc3, dec4], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec3], dim=1))
        dec1 = self.dec1(torch.cat([enc1, dec2], dim=1))

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
    set_determinism(seed=1010710)   # 100 and 1010710

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    metric = MonaiDiceMetric(include_background=True, reduction="mean")

    # Load Net
    net = LatentImages(
        channels=(64, 128, 256, 512, 1024),
    ).to(device)

    net.load_state_dict(torch.load(
        r"C:\Users\Labo\Documents\GitHub\ProstateCancerPrognosisAI\src\models\torch\segmentation\net\prostate_net\saved_parameters\best_parameters_avg.pt"))

    net.eval()

    # Task
    task = SegmentationTask(
        criterion=DiceLoss(),
        organ="Prostate",
        modality="CT"
    )

    # Database
    database = PatientsDatabase(
        path_to_database=r"C:\Users\Labo\Documents\GitHub\ProstateCancerPrognosisAI\applications\local_data\learning_set.h5"
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
            true_seg = patient.y["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

            layers = net.get_latent_images(img)

            out = torch.sigmoid(layers.dec1)
            out = torch.round(out).cpu()

            print(metric(y_pred=out, y=true_seg))

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

    ax["F"].imshow(layers.bottom.cpu()[0][0][int(size / 32), 1:-1, 1:-1], cmap="Greys_r")
    ax["F"].set_title(f"bottom {np.shape(layers.bottom)}")

    ax["G"].imshow(layers.dec4.cpu()[0][0][int(size / 16), :, :], cmap="Greys_r")
    ax["G"].set_title(f"dec4 {np.shape(layers.dec4)}")

    ax["H"].imshow(layers.dec3.cpu()[0][0][int(size / 8), :, :], cmap="Greys_r")
    ax["H"].set_title(f"dec3 {np.shape(layers.dec3)}")

    ax["I"].imshow(layers.dec2.cpu()[0][4][int(size / 4), :, :], cmap="Greys_r")
    ax["I"].set_title(f"dec2 {np.shape(layers.dec2)}")

    ax["J"].imshow(layers.dec1.cpu()[0][0][int(size / 2), :, :], cmap="Greys_r")
    ax["J"].set_title(f"dec1 {np.shape(layers.dec1)}")

    ax["K"].imshow(out.cpu()[0][0][int(size / 2), :, :], cmap="Greys_r")
    ax["K"].set_title(f"out {np.shape(out)}")

    plt.show()

    # Show layers statistics
    if show_layer_stats:
        # Enc1
        print("----------------enc1-----------------")
        print('max', )
        #channels = np.shape(layers.enc1)[1]


    # Save Latent Images
    if save_latent_images:

        c: int = 0
        l: str = "img"
        plt.figure()
        plt.imshow(layers.img.cpu()[0][c][int(size / 2), :, :], cmap="Greys_r")
        plt.axis("off")
        plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.show()

        l: str = "enc1"
        channels = [0, 4, 11, 32, 25, 37, 52, 63]
        for c in channels:
            plt.figure()
            plt.imshow(layers.enc1.cpu()[0][c][int(size / 4), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "enc2"
        channels = [0, 4, 11, 32, 36, 121, 59, 47]
        for c in channels:
            plt.figure()
            plt.imshow(layers.enc2.cpu()[0][c][int(size / 8), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "enc3"
        channels = [0, 4, 11, 32, 49, 244, 3, 224]
        for c in channels:
            plt.figure()
            plt.imshow(layers.enc3.cpu()[0][c][int(size / 16), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "enc4"
        channels = [0, 4, 11, 32, 446, 207, 87, 482]
        for c in channels:
            plt.figure()
            plt.imshow(layers.enc4.cpu()[0][c][int(size / 32), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "bottom"
        channels = [0, 4, 11, 32, 856, 528, 184, 811]
        for c in channels:
            plt.figure()
            plt.imshow(layers.bottom.cpu()[0][c][int(size / 32), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "dec4"
        channels = [0, 4, 11, 32, 146, 81, 21, 247]
        for c in channels:
            plt.figure()
            plt.imshow(layers.dec4.cpu()[0][c][int(size / 16), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "dec3"
        channels = [0, 4, 11, 32, 68, 39, 21, 55]
        for c in channels:
            plt.figure()
            plt.imshow(layers.dec3.cpu()[0][c][int(size / 8), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        l: str = "dec2"
        channels = [0, 4, 11, 32, 62, 7, 30, 19]
        for c in channels:
            plt.figure()
            plt.imshow(layers.dec2.cpu()[0][c][int(size / 4), :, :], cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.show()

        c: int = 0
        l: str = "dec1"
        plt.figure()
        plt.imshow(layers.dec1.cpu()[0][c][int(size / 2), :, :], cmap="Greys_r")
        plt.axis("off")
        plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.show()

        c: int = 0
        l: str = "out"
        plt.figure()
        plt.imshow(out.cpu()[0][c][int(size / 2), :, :], cmap="Greys_r")
        plt.axis("off")
        plt.savefig(f"C:/Users/Labo/Desktop/latent_images_png/{l}_{c}.png", dpi=600, bbox_inches="tight", pad_inches=0)
        plt.show()



    # # View a Single Latent Image
    # patient_idx: int = 0                                # which patient to visualize
    # ImageViewer().view_latent_image(layers.enc1.cpu()[patient_idx])
    #
    # # View weights
    # net_children = list(net.children())                 # net_children[i], i from (0 = enc1, 1 = enc2, 2 = enc3, ...)
    # ImageViewer().view_filter(net_children[2].conv1.weight.cpu())
