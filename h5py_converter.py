from delia.databases import PatientsDatabase
from delia.extractors import PatientsDataExtractor
import matplotlib.pyplot as plt
import numpy as np
import h5py
from delia.transforms import ResampleD, MatchingResampleD
from monai.transforms import (
    CenterSpatialCropD,
    Compose,
    ScaleIntensityD,
    ThresholdIntensityD,
    GaussianSharpenD,
    Rotate90D
)

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

keys = ["CT"]
series = {"CT": [
            "THORAX 1.0 B45f",
            "CHEST 1.25 MM",
            "LUNG WINDOW",
            "THIN LUNG WINDOW",
            "Thorax  1.0  B45f",
            ".625 mm Chest",
            "THINS",
            "Recon 2: CHEST",
            "Recon 3: CT CHEST W/O",
            "0.625  DMPR On + SS50",
            "Recon 2: CAP",
            "1.25MM CHEST BONE",
            "CHEST 1.0  B45f",
            "LUNG 1MM B45f",
            "1.25MM CHEST BONE PLUS",
            "CT Thick Axials 2.5mm",
            "0.625MM CHEST NO PACS",
            "CHEST LUNG",
            "THIN CHEST LUNG",
            "SUPER",
            "CHEST NON-CON",
            "AX 1.25",
            "THIN",
            "IN REACH",
            "CHEST 1.25MM",
            "LUNG",
            "Chest wo  3.0  B40f"
        ]}
segs_keys = ["Heart", "Segmentation", "Tissue"]
need_transform = True
graph_scans = False

if need_transform is None:
    pass
elif need_transform:
    # transform_list = [
    #             ResampleD(keys=keys, out_spacing=(1.5, 1.5, 1.5)),
    #             MatchingResampleD(reference_image_key="CT", matching_keys=["Heart", "Segmentation", "Tissue"]),
    #             CenterSpatialCropD(keys=["CT", "Heart", "Segmentation", "Tissue"], roi_size=(192, 192, 192)),
    #             ScaleIntensityD(keys=keys, minv=-1, maxv=1)]
    # patients_data_extractor = PatientsDataExtractor(
    #     path_to_patients_folder=
    #     "/Users/felixdesroches/Desktop/Stages et notes/Stage -E23/brain_segmentation_E23/lung_data/manifest-1685649685383/patients",
    #     series_descriptions=series,
    #     transforms=Compose(transform_list))
    # database = PatientsDatabase(path_to_database="patients_database_lung.h5")
    # database.create(
    #     patients_data_extractor=patients_data_extractor,
    #     overwrite_database=True,
    #     tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    # )
    # database.close()
    with h5py.File("patients_database_lung.h5", "r+") as f:
        for key in list(f.keys()):
            group = f[key]["0"]["0"]
            group.move(list(group.keys())[0], "Lung")
else:
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=
            "/Users/felixdesroches/Desktop/Stages et notes/Stage -E23/brain_segmentation_E23/lung_data/manifest-1685649685383/patients",
        series_descriptions=series
    )
    database = PatientsDatabase(path_to_database="patients_database.h5")
    database.create(
        patients_data_extractor=patients_data_extractor,
        overwrite_database=True,
        tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    )


if graph_scans:
    filename = 'patients_database_2.h5'
    for i in range(8):
        with h5py.File(filename, "r") as f:
            patient_key = list(f.keys())[i]
            patient_data = f[patient_key]["0"]["0"]["Lung"]
            data_4 = np.array(patient_data)
            print(data_4.shape)

        precision = 2
        subplot_x = 10
        subplot_y = (data_4.shape[2] // precision) // subplot_x
        fig, arr = plt.subplots(subplot_y, subplot_x)

        for i in range(subplot_y * subplot_x):
            if precision * i > data_4.shape[2]:
                continue
            arr[i % subplot_y, i // subplot_y].imshow(data_4[:, :, i * precision], cmap="gray")
            arr[i % subplot_y, i // subplot_y].axis("off")
        plt.show()
