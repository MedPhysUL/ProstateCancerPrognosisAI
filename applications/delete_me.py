"""
    @file:              08_train_unet.py
    @Author:            Maxence Larose

    @Creation Date:     04/2023
    @Last modification: 04/2023

    @Description:       This script is used to train a unet model.
"""

import env_apps
from enum import IntEnum
import os

from delia.databases import PatientsDatabase

from constants import *
from src.data.datasets import ImageDataset, ProstateCancerDataset
from monai.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np


# """
#     @file:              create_database.py
#     @Author:            Maxence Larose
#
#     @Creation Date:     06/2022
#     @Last modification: 11/2022
#
#     @Description:       This file shows how to create an HDF5 file database using DELIA.
# """
#
# import env_apps
#
# from delia.databases import PatientsDatabase
# from delia.extractors import PatientsDataExtractor
# from delia.transforms import (
#     MatchingCentroidSpatialCropD,
#     MatchingCropForegroundD,
#     MatchingResampleD,
#     PETtoSUVD,
#     ResampleD
# )
# from monai.transforms import (
#     CenterSpatialCropD,
#     Compose,
#     KeepLargestConnectedComponentD,
#     ScaleIntensityRangeD,
#     SpatialCropD
# )
#
#
# if __name__ == "__main__":
#     # ----------------------------------------------------------------------------------------------------------- #
#     #                                               Logs Setup                                                    #
#     # ----------------------------------------------------------------------------------------------------------- #
#     env_apps.configure_logging("logging_conf.yaml")
#
#     # ----------------------------------------------------------------------------------------------------------- #
#     #                                               Transforms                                                    #
#     # ----------------------------------------------------------------------------------------------------------- #
#     automatic_crop = [
#         SpatialCropD(keys=["CT", "PT", "Prostate"], roi_slices=[slice(50, 210), slice(None), slice(None)]),
#         KeepLargestConnectedComponentD(keys=["Prostate"]),
#         CenterSpatialCropD(keys=["CT", "PT", "Prostate"], roi_size=(1000, 160, 160)),
#     ]
#
#     ideal_crop = [
#         SpatialCropD(keys=["CT", "PT", "Prostate"], roi_slices=[slice(30, 700), slice(None), slice(None)]),
#         KeepLargestConnectedComponentD(keys=["Prostate"]),
#         MatchingCentroidSpatialCropD(segmentation_key="Prostate", matching_keys=["CT", "PT"], roi_size=(160, 160, 160))
#     ]
#
#     transforms = Compose(
#         [
#             ResampleD(keys=["CT"], out_spacing=(1, 1, 1)),
#             MatchingResampleD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
#             MatchingCropForegroundD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
#             *ideal_crop,
#             PETtoSUVD(keys=["PT"]),
#             # ScaleIntensityRangeD(keys=["CT"], a_min=-200, a_max=200, b_min=0, b_max=1, clip=True),
#             ScaleIntensityRangeD(keys=["PT"], a_min=0, a_max=25, b_min=0, b_max=20, clip=True)
#         ]
#     )
#
#     # ----------------------------------------------------------------------------------------------------------- #
#     #                                      Create patients data extractor                                         #
#     # ----------------------------------------------------------------------------------------------------------- #
#     patients_data_extractor = PatientsDataExtractor(
#         path_to_patients_folder=r"local_data/Holdout_set",
#         series_descriptions=r"local_data/series_descriptions.json",
#         transforms=transforms
#     )
#
#     # ----------------------------------------------------------------------------------------------------------- #
#     #                                                Create database                                              #
#     # ----------------------------------------------------------------------------------------------------------- #
#     database = PatientsDatabase(path_to_database=r"local_data/holdout_set.h5")
#
#     database.create(
#         patients_data_extractor=patients_data_extractor,
#         organs_to_keep="Prostate",
#         tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
#     )



def plot(img, title):
    fig, axes = plt.subplots(1, 1)
    img_max = np.max(img)
    img_min = np.min(img)
    img_axial = np.moveaxis(img, 2, 0)

    if img_axial.shape != (96, 96, 96):
        raise AssertionError("Size mismatch.")

    axes.imshow(
        np.fliplr(np.rot90(np.rot90(img_axial[int(img_axial.shape[0]/2)]))),
        cmap="Greys_r",
        vmax=img_max,
        vmin=img_min
    )
    axes.set_axis_off()
    axes.set_title(title)


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=r"local_data/holdout_set.h5")

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT", "PT"},
        transforms=Compose([])
    )

    for i, images in enumerate(image_dataset):
        patient_id = image_dataset._database[i].name[1:]

        PATH = r"C:\Users\Labo\Downloads\test\CT"
        plot(images["CT"], patient_id)
        plt.savefig(fname=os.path.join(PATH, patient_id), dpi=150, bbox_inches='tight')
        # plt.show()
        plt.close()

        PATH = r"C:\Users\Labo\Downloads\test\PT"
        plot(images["PT"], patient_id)
        plt.savefig(fname=os.path.join(PATH, patient_id), dpi=150, bbox_inches='tight')
        # plt.show()
        plt.close()
