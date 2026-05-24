"""
segmentation_dataset.py

Dataset download utilities and PyTorch Dataset definition for the
MO437 semantic segmentation coursework dataset.

This file contains:
    - Dataset path constants.
    - A download/extract helper.
    - A dataset-info loader.
    - The SegmentationDataset class.

Expected dataset structure after extraction:
    data/
        segmentation_640x360/
            dataset_info.json
            images/
                train/
                val/
                test/
            masks/
                train/
                val/
                test/
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------
# Dataset location and download configuration.
# ---------------------------------------------------------------------

# Default folder where all data files will be stored.
DATA_FOLDER = Path.cwd() / "data"

# Name of the downloaded zip file.
SEG_DATASET_ZIPFILE = DATA_FOLDER / "segmentation_640x360.zip"

# Name of the extracted dataset folder.
SEG_DATASET_FOLDER = DATA_FOLDER / "segmentation_640x360"

# Remote URL used in the original notebook.
SEG_DATASET_URL = (
    "https://www.ic.unicamp.br/~edson/disciplinas/mo437/2026-1s/"
    "course-work-dataset/segmentation_640x360.zip"
)


def download_and_extract_segmentation_dataset(
    data_folder: str | os.PathLike = DATA_FOLDER,
    dataset_url: str = SEG_DATASET_URL,
    zip_filename: str = "segmentation_640x360.zip",
    extracted_folder_name: str = "segmentation_640x360",
) -> Path:
    """
    Download and extract the segmentation dataset if it is not already present.

    Parameters
    ----------
    data_folder:
        Folder where the zip file and extracted dataset folder should live.

    dataset_url:
        URL from which the dataset zip file will be downloaded.

    zip_filename:
        Name used for the downloaded zip file.

    extracted_folder_name:
        Name of the top-level folder expected inside the zip file.

    Returns
    -------
    Path
        Path to the extracted dataset folder.
    """

    # Convert the data folder argument into a pathlib Path object.
    data_folder = Path(data_folder)

    # Build the full local path for the zip file.
    zip_path = data_folder / zip_filename

    # Build the full local path for the extracted dataset folder.
    extracted_folder = data_folder / extracted_folder_name

    # Ensure the parent data folder exists before downloading anything.
    data_folder.mkdir(parents=True, exist_ok=True)

    # Download the dataset zip only if it is not already available locally.
    if not zip_path.is_file():
        print(f"Downloading dataset from {dataset_url}")
        print(f"Saving zip file to {zip_path}")
        urllib.request.urlretrieve(dataset_url, zip_path)
    else:
        print(f"Zip file already exists: {zip_path}")

    # Extract the dataset only if the extracted folder is not already available.
    if not extracted_folder.is_dir():
        print(f"Extracting dataset into {data_folder}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member_name in zip_ref.namelist():
                # Extract only the expected dataset folder to avoid unrelated files.
                if member_name.startswith(extracted_folder_name):
                    zip_ref.extract(member_name, data_folder)
    else:
        print(f"Extracted dataset already exists: {extracted_folder}")

    # Return the path so other scripts can use it directly.
    return extracted_folder


def load_segmentation_dataset_config(dataset_folder: str | os.PathLike) -> dict:
    """
    Load the dataset_info.json file for the segmentation dataset.

    Parameters
    ----------
    dataset_folder:
        Path to the extracted segmentation dataset folder.

    Returns
    -------
    dict
        Parsed JSON dictionary containing class names, colors, and metadata.
    """

    # Convert to Path so path operations are robust across operating systems.
    dataset_folder = Path(dataset_folder)

    # Build the path to the JSON metadata file.
    config_path = dataset_folder / "dataset_info.json"

    # Raise a clear error if the dataset has not been downloaded/extracted yet.
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Could not find dataset configuration file: {config_path}. "
            "Run download_and_extract_segmentation_dataset() first."
        )

    # Open and parse the JSON file.
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for semantic segmentation.

    Each item returned by this dataset is:
        image, one_hot_mask

    where:
        - image has shape [3, H, W].
        - one_hot_mask has shape [num_valid_classes, H, W].

    The original masks use integer class IDs:
        0 = background / ignored class
        1..num_valid_classes = active segmentation classes

    Therefore, the one-hot mask excludes class 0 and maps:
        mask value 1 -> channel 0
        mask value 2 -> channel 1
        ...
        mask value num_valid_classes -> channel num_valid_classes - 1
    """

    def __init__(
        self,
        images_dir: str | os.PathLike,
        masks_dir: str | os.PathLike,
        num_valid_classes: int,
        ignore_index: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        images_dir:
            Folder containing RGB input images.

        masks_dir:
            Folder containing grayscale segmentation masks.

        num_valid_classes:
            Number of foreground classes to encode in the one-hot mask.

        ignore_index:
            Class index to ignore. The current one-hot conversion ignores class 0
            by default, matching the original notebook logic.

        transform:
            Optional callable that receives and returns an ``(image, mask)`` pair.
            Image is a PIL RGB image before ToTensor, and mask is a torch LongTensor
            containing integer class IDs.
        """

        # Store the folder containing input images.
        self.images_dir = Path(images_dir)

        # Store the folder containing segmentation masks.
        self.masks_dir = Path(masks_dir)

        # Store the number of valid non-background classes.
        self.num_valid_classes = int(num_valid_classes)

        # Store the ignored class index for clarity and future extension.
        self.ignore_index = int(ignore_index)

        # Store the optional transform pipeline.
        self.transform = transform

        # Validate image and mask folders early to produce readable errors.
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")

        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks folder not found: {self.masks_dir}")

        # Sort file names to make dataset ordering deterministic.
        self.images = sorted(os.listdir(self.images_dir))

    def __len__(self) -> int:
        """Return the number of image/mask pairs in the dataset."""

        # The dataset length is the number of image files.
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load one image/mask pair and return tensors ready for training.

        Parameters
        ----------
        idx:
            Integer sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The transformed image tensor and one-hot encoded mask tensor.
        """

        # Get the file name associated with the requested index.
        image_name = self.images[idx]

        # Build the full path to the input RGB image.
        img_path = self.images_dir / image_name

        # Build the full path to the corresponding mask.
        mask_path = self.masks_dir / image_name

        # Load the input image as RGB.
        image = Image.open(img_path).convert("RGB")

        # Load the mask as grayscale to preserve integer class IDs.
        mask = Image.open(mask_path).convert("L")

        # Convert the mask to a NumPy integer array first.
        # This avoids torchvision converting integer class IDs into [0, 1] floats.
        mask_array = np.array(mask, dtype=np.int64)

        # Convert the NumPy mask array into a torch tensor of class IDs.
        mask_tensor = torch.from_numpy(mask_array)

        # Apply paired image/mask transformations if a transform pipeline exists.
        if self.transform is not None:
            image, mask_tensor = self.transform((image, mask_tensor))

            # Ensure the mask remains integer class IDs after transformations.
            mask_tensor = mask_tensor.long()
        else:
            # If no transform is given, convert only the image to a tensor.
            image = TF.to_tensor(image)

        # Read the transformed mask spatial dimensions.
        height, width = mask_tensor.shape

        # Allocate the one-hot mask tensor, excluding the ignored/background class.
        one_hot_mask = torch.zeros(
            size=(self.num_valid_classes, height, width),
            dtype=torch.float32,
        )

        # Fill one channel per valid foreground class.
        for class_id in range(1, self.num_valid_classes + 1):
            # Convert class_id into zero-based channel index.
            channel_idx = class_id - 1

            # Mark pixels belonging to the current class.
            one_hot_mask[channel_idx] = mask_tensor == class_id

        # Return the image tensor and the one-hot segmentation target.
        return image, one_hot_mask
