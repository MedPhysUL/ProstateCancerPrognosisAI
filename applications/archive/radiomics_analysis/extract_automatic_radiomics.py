
import env_apps

import os

from delia.databases import PatientsDatabase
from delia.radiomics import RadiomicsDataset, RadiomicsFeatureExtractor
from monai.data import DataLoader
import SimpleITK as sitk
import torch

from constants import (
    AUTOMATIC_RADIOMICS_MODELS_PATH,
    AUTOMATIC_EXTRACTED_RADIOMICS_PATH,
    CT_FEATURES_EXTRACTOR_PARAMS_PATH,
    LEARNING_SET_PATH,
    MASKS_PATH,
    PROSTATE_SEGMENTATION_TASK,
    PT_FEATURES_EXTRACTOR_PARAMS_PATH,
    SEED
)
from src.data.datasets import ImageDataset, ProstateCancerDataset
from src.data.processing.sampling import extract_masks, Mask
from src.models.torch.segmentation import Unet


def save_radiomics(dataset: ProstateCancerDataset, radiomics_dataset: RadiomicsDataset, modality: str):
    radiomics_dataset.extractor = RadiomicsFeatureExtractor(
        path_to_params=CT_FEATURES_EXTRACTOR_PARAMS_PATH if modality == "CT" else PT_FEATURES_EXTRACTOR_PARAMS_PATH,
        geometryTolerance=1e-4
    )

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=None)
    radiomics_features = {}
    for idx, (features, _) in enumerate(data_loader):
        pred = model.predict(features=features, probability=False)
        mask = pred["SegmentationTask('modality'='CT', 'organ'='Prostate')"]

        radiomics = radiomics_dataset.extractor.execute(
            imageFilepath=sitk.GetImageFromArray(
                arr=features.image[f"{modality}"].cpu().detach().numpy()[0, 0, ...],
                isVector=False
            ),
            maskFilepath=sitk.GetImageFromArray(
                arr=mask.cpu().detach().numpy()[0, 0, ...],
                isVector=False
            ),
            label=None,
            label_channel=None,
            voxelBased=False
        )

        radiomics_features[f"{idx}"] = radiomics

    radiomics_dataset.save(radiomics_features)


if __name__ == '__main__':
    database = PatientsDatabase(path_to_database=LEARNING_SET_PATH)

    image_dataset = ImageDataset(
        database=database,
        modalities={"CT", "PT"},
        tasks=PROSTATE_SEGMENTATION_TASK
    )

    dataset = ProstateCancerDataset(image_dataset=image_dataset, table_dataset=None)

    masks = extract_masks(MASKS_PATH, k=5, l=5)
    for k in range(5):
        model_state = torch.load(
            os.path.join(AUTOMATIC_RADIOMICS_MODELS_PATH, f"outer_split_{k}", "model.pth")
        )["model_state"]

        dataset.update_masks(
            train_mask=masks[k][Mask.TRAIN],
            test_mask=masks[k][Mask.TEST],
            valid_mask=masks[k][Mask.VALID]
        )

        model = Unet(
            image_keys="CT",
            spatial_dims=3,
            num_res_units=3,
            dropout=0.2,
            device=torch.device("cuda"),
            seed=SEED
        ).build(dataset)

        model.load_state_dict(model_state)

        radiomics_dataset = RadiomicsDataset(
            path_to_dataset=os.path.join(AUTOMATIC_EXTRACTED_RADIOMICS_PATH, f"ct_outer_split_{k}.csv")
        )
        save_radiomics(dataset, radiomics_dataset, "CT")

        radiomics_dataset = RadiomicsDataset(
            path_to_dataset=os.path.join(AUTOMATIC_EXTRACTED_RADIOMICS_PATH, f"pt_outer_split_{k}.csv")
        )
        save_radiomics(dataset, radiomics_dataset, "PT")
