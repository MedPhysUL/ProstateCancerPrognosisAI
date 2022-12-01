"""
    @file:              create_database.py
    @Author:            Maxence Larose

    @Creation Date:     06/2022
    @Last modification: 11/2022

    @Description:       This file shows how to create an HDF5 file database using DELIA.
"""

import env_apps

from delia.databases import PatientsDatabase
from delia.extractors import PatientsDataExtractor
from delia.transforms import (
    CopySegmentationsD,
    MatchingCropForegroundD,
    PETtoSUVD,
    ResampleD
)
from monai.transforms import (
    CenterSpatialCropD,
    Compose,
    KeepLargestConnectedComponentD,
    ScaleIntensityD,
    SpatialCropD,
    ThresholdIntensityD,
)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Logs Setup                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    env_apps.configure_logging("logging_conf.yaml")

    # ----------------------------------------------------------------------------------------------------------- #
    #                                      Create patients data extractor                                         #
    # ----------------------------------------------------------------------------------------------------------- #
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=r"local_data/Learning_set",
        series_descriptions=r"local_data/series_descriptions.json",
        transforms=Compose(
            [
                ResampleD(keys=["CT", "PET", "Prostate"], out_spacing=(1.5, 1.5, 1.5)),
                MatchingCropForegroundD(image_key="CT", matching_keys=["PET", "Prostate"]),
                SpatialCropD(keys=["CT", "PET", "Prostate"], roi_slices=[slice(50, 210), slice(None), slice(None)]),
                CenterSpatialCropD(keys=["CT", "PET", "Prostate"], roi_size=(1000, 160, 160)),
                PETtoSUVD(keys=["PET"]),
                ThresholdIntensityD(keys=["CT"], threshold=-250, above=True, cval=-250),
                ThresholdIntensityD(keys=["CT"], threshold=500, above=False, cval=500),
                ScaleIntensityD(keys=["CT"], minv=0, maxv=1),
                KeepLargestConnectedComponentD(keys=["Prostate"]),
                CopySegmentationsD(segmented_image_key="CT", unsegmented_image_key="PET"),
            ]
        )
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #                                                Create database                                              #
    # ----------------------------------------------------------------------------------------------------------- #
    database = PatientsDatabase(path_to_database=r"local_data/learning_set.h5")

    database.create(
        patients_data_extractor=patients_data_extractor,
        organs_to_keep="Prostate",
        tags_to_use_as_attributes=[(0x0008, 0x103E), (0x0020, 0x000E), (0x0008, 0x0060)]
    )
