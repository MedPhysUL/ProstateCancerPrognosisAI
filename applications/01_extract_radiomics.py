"""
    @file:              01_extract_radiomics.py
    @Author:            Maxence Larose

    @Creation Date:     11/2022
    @Last modification: 06/2023

    @Description:       This file shows how to extract radiomics from CT and PET using DELIA.
"""

import env_apps

import os

from delia.extractors import PatientsDataExtractor
from delia.radiomics import RadiomicsDataset, RadiomicsFeatureExtractor
from delia.transforms import (
    MatchingCentroidSpatialCropD,
    MatchingCropForegroundD,
    MatchingResampleD,
    CopySegmentationsD,
    PETtoSUVD,
    ResampleD
)
from monai.transforms import (
    Compose,
    KeepLargestConnectedComponentD,
    ScaleIntensityRangeD,
    SpatialCropD
)

from constants import (
    CT_FEATURES_EXTRACTOR_PARAMS_PATH,
    LEARNING_SET_FOLDER_PATH,
    LOGGING_CONFIG_PATH,
    MANUAL_EXTRACTED_RADIOMICS_PATH,
    PT_FEATURES_EXTRACTOR_PARAMS_PATH,
    SERIES_DESCRIPTIONS_PATH
)


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Logs Setup                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    env_apps.configure_logging(LOGGING_CONFIG_PATH)

    # ----------------------------------------------------------------------------------------------------------- #
    #                                               Transforms                                                    #
    # ----------------------------------------------------------------------------------------------------------- #
    transforms = Compose([
        ResampleD(keys=["CT"], out_spacing=(1.0, 1.0, 1.0)),
        MatchingResampleD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
        MatchingCropForegroundD(reference_image_key="CT", matching_keys=["PT", "Prostate"]),
        SpatialCropD(keys=["CT", "PT", "Prostate"], roi_slices=[slice(30, 740), slice(None), slice(None)]),
        KeepLargestConnectedComponentD(keys=["Prostate"]),
        MatchingCentroidSpatialCropD(segmentation_key="Prostate", matching_keys=["CT", "PT"], roi_size=(128, 128, 128)),
        PETtoSUVD(keys=["PT"]),
        ScaleIntensityRangeD(keys=["CT"], a_min=-200, a_max=250, b_min=0, b_max=1, clip=True),
        ScaleIntensityRangeD(keys=["PT"], a_min=0, a_max=25, b_min=0, b_max=1, clip=True),
        CopySegmentationsD(segmented_image_key="CT", unsegmented_image_key="PT")
    ])

    # ----------------------------------------------------------------------------------------------------------- #
    #                                      Create patients data extractor                                         #
    # ----------------------------------------------------------------------------------------------------------- #
    patients_data_extractor = PatientsDataExtractor(
        path_to_patients_folder=LEARNING_SET_FOLDER_PATH,
        tag="SeriesDescription",
        tag_values=SERIES_DESCRIPTIONS_PATH,
        transforms=transforms
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #    Extract radiomics features of the CT image from the segmentation of the prostate made on the CT image    #
    # ----------------------------------------------------------------------------------------------------------- #
    ct_radiomics_dataset = RadiomicsDataset(
        path_to_dataset=os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_ct_radiomics.csv")
    )

    ct_radiomics_dataset.extractor = RadiomicsFeatureExtractor(
        path_to_params=CT_FEATURES_EXTRACTOR_PARAMS_PATH,
        geometryTolerance=1e-4
    )

    ct_radiomics_dataset.create(
        patients_data_extractor=patients_data_extractor,
        organ="Prostate",
        image_modality="CT"
    )

    # ----------------------------------------------------------------------------------------------------------- #
    #    Extract radiomics features of the PT image from the segmentation of the prostate made on the CT image    #
    # ----------------------------------------------------------------------------------------------------------- #
    pt_radiomics_dataset = RadiomicsDataset(
        path_to_dataset=os.path.join(MANUAL_EXTRACTED_RADIOMICS_PATH, "learning_pt_radiomics.csv")
    )

    pt_radiomics_dataset.extractor = RadiomicsFeatureExtractor(
        path_to_params=PT_FEATURES_EXTRACTOR_PARAMS_PATH,
        geometryTolerance=1e-4
    )

    pt_radiomics_dataset.create(
        patients_data_extractor=patients_data_extractor,
        organ="Prostate",
        image_modality="PT"
    )
