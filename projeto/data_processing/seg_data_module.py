#datamodule

"""
seg_data_module.py

PyTorch Lightning DataModule for the segmentation dataset.

This file integrates:
    - SegmentationDataset from segmentation_dataset.py
    - transform classes from transforms.py

Typical usage
-------------
from segmentation_dataset import (
    download_and_extract_segmentation_dataset,
    load_segmentation_dataset_config,
)

from seg_data_module import SegmentationLightningDataModule

dataset_folder = download_and_extract_segmentation_dataset()
dataset_config = load_segmentation_dataset_config(dataset_folder)

data_module = SegmentationLightningDataModule(
    data_dir=dataset_folder,
    num_valid_classes=dataset_config["num_classes"],
    batch_size=16,
    num_workers=8,
)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import lightning as pl
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data_processing.segmentation_dataset import SegmentationDataset
from data_processing.seg_transforms import (
    SegmentationImageTransformOnly,
    SegmentationRandomFlip,
    SegmentationRandomResizedCrop,
    SegmentationRandomRotation,
)


class SegmentationLightningDataModule(pl.LightningDataModule):
    """
    LightningDataModule for semantic segmentation.

    This class creates:
        - train dataset and dataloader
        - validation dataset and dataloader
        - test dataset and dataloader

    It also centralizes the training/validation/test transforms so the training
    script does not need to manually instantiate datasets.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike,
        num_valid_classes: int,
        batch_size: int = 16,
        num_workers: int = 2,
        image_size: tuple[int, int] = (352, 640),
        train_mean: Sequence[float] = (0.3204, 0.3455, 0.3181),
        train_std: Sequence[float] = (0.2326, 0.2344, 0.2466),
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
    ) -> None:
        """
        Initialize the segmentation DataModule.

        Parameters
        ----------
        data_dir:
            Path to the extracted dataset folder containing images/ and masks/.

        num_valid_classes:
            Number of foreground classes used for one-hot target masks.

        batch_size:
            Number of samples per batch.

        num_workers:
            Number of DataLoader worker processes.

        image_size:
            Final spatial size as (height, width).

        train_mean:
            Channel-wise mean used for normalization.

        train_std:
            Channel-wise standard deviation used for normalization.

        pin_memory:
            Whether DataLoader should pin memory. Usually useful with GPU training.

        persistent_workers:
            Whether DataLoader workers should stay alive between epochs.
            If None, this is automatically enabled only when num_workers > 0.
        """

        # Initialize LightningDataModule internals.
        super().__init__()

        # Store the dataset root path.
        self.data_dir = Path(data_dir)

        # Store number of valid foreground classes.
        self.num_valid_classes = int(num_valid_classes)

        # Store DataLoader batch size.
        self.batch_size = int(batch_size)

        # Store DataLoader worker count.
        self.num_workers = int(num_workers)

        # Store the target image size for all splits.
        self.image_size = image_size

        # Store normalization statistics.
        self.train_mean = list(train_mean)
        self.train_std = list(train_std)

        # Store whether DataLoader should use pinned memory.
        self.pin_memory = bool(pin_memory)

        # Enable persistent workers only when workers exist, unless user overrides.
        if persistent_workers is None:
            self.persistent_workers = self.num_workers > 0
        else:
            self.persistent_workers = bool(persistent_workers)

        # Build the stochastic training transform pipeline.
        self.train_transform = T.Compose(
            [
                # Randomly rotate image and mask together.
                SegmentationRandomRotation(degrees=15),

                # Randomly crop and resize image and mask together.
                SegmentationRandomResizedCrop(
                    resize_size=self.image_size,
                    scale=(0.5, 1.0),
                    ratio=(3.0 / 4.0, 4.0 / 3.0),
                ),

                # Randomly flip image and mask together.
                SegmentationRandomFlip(prob=0.5),

                # Convert only the image to tensor; keep mask as integer labels.
                SegmentationImageTransformOnly(T.ToTensor()),

                # Normalize only the image tensor.
                SegmentationImageTransformOnly(
                    T.Normalize(mean=self.train_mean, std=self.train_std)
                ),
            ]
        )

        # Build the deterministic validation/test transform pipeline.
        self.val_test_transform = T.Compose(
            [
                # Resize image and mask to the same final size without random crop.
                SegmentationRandomResizedCrop(
                    resize_size=self.image_size,
                    scale=(1.0, 1.0),
                    ratio=(1.0, 1.0),
                ),

                # Convert only the image to tensor.
                SegmentationImageTransformOnly(T.ToTensor()),

                # Normalize only the image tensor.
                SegmentationImageTransformOnly(
                    T.Normalize(mean=self.train_mean, std=self.train_std)
                ),
            ]
        )

        # These attributes are initialized in setup().
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _make_dataset(self, split: str, transform) -> SegmentationDataset:
        """
        Create a SegmentationDataset for a given split.

        Parameters
        ----------
        split:
            Dataset split name: "train", "val", or "test".

        transform:
            Transform pipeline to apply to the split.

        Returns
        -------
        SegmentationDataset
            Dataset instance for the requested split.
        """

        # Build path to the split image folder.
        images_dir = self.data_dir / "images" / split

        # Build path to the split mask folder.
        masks_dir = self.data_dir / "masks" / split

        # Create and return the dataset.
        return SegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            num_valid_classes=self.num_valid_classes,
            transform=transform,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets for the requested Lightning stage.

        Lightning calls this automatically before training/testing.
        """

        # Create train and validation datasets for fitting.
        if stage == "fit" or stage is None:
            self.train_ds = self._make_dataset(
                split="train",
                transform=self.train_transform,
            )

            self.val_ds = self._make_dataset(
                split="val",
                transform=self.val_test_transform,
            )

        # Create test dataset for testing.
        if stage == "test" or stage is None:
            self.test_ds = self._make_dataset(
                split="test",
                transform=self.val_test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training DataLoader.

        Training uses shuffle=True because sample order should change each epoch.
        """

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation DataLoader.

        Validation uses shuffle=False for deterministic evaluation.
        """

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the test DataLoader.

        Testing uses shuffle=False for deterministic evaluation.
        """

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
