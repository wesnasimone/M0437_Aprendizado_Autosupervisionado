"""
transforms.py

Image and image/mask transformation utilities for semantic segmentation.

This file contains:
    - compute_mean_std
    - SegmentationImageTransformOnly
    - SegmentationRandomResizedCrop
    - SegmentationRandomRotation
    - SegmentationRandomFlip

The paired transforms are designed for inputs of the form:
    (image, mask)

where:
    - image is usually a PIL image before ToTensor.
    - mask is a torch tensor containing integer class IDs.
"""

from __future__ import annotations

import random
from typing import Callable, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.transforms import v2


def compute_mean_std(img_dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and standard deviation for an image dataset.

    Parameters
    ----------
    img_dataset:
        Dataset returning ``(image, label)`` pairs, where image is a tensor with
        shape [C, H, W].

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mean and standard deviation tensors with shape [C].
    """

    # Initialize the running sum for each image channel.
    mean = 0.0

    # Initialize the total number of pixels counted per channel.
    count = 0

    # First pass: compute the channel-wise mean.
    for image, _ in img_dataset:
        # Flatten each channel into a single dimension: [C, H * W].
        flatten_img = torch.flatten(image, start_dim=1)

        # Add the sum of all pixels for each channel.
        mean += flatten_img.sum(dim=1)

        # Add the number of pixels in each channel.
        count += flatten_img.size(1)

    # Divide the accumulated sum by the number of pixels.
    mean = mean / count

    # Initialize the running squared-difference sum.
    std = 0.0

    # Second pass: compute the channel-wise variance.
    for image, _ in img_dataset:
        # Flatten each channel into a single dimension: [C, H * W].
        flatten_img = torch.flatten(image, start_dim=1)

        # Move channels to the last dimension so broadcasting with mean is clear.
        centered_img = flatten_img.permute(1, 0) - mean

        # Square the difference from the mean.
        squared_diff = torch.pow(centered_img, 2)

        # Accumulate squared differences per channel.
        std += squared_diff.sum(dim=0)

    # Convert variance into standard deviation.
    std = torch.pow(std / count, 0.5)

    # Return channel-wise mean and standard deviation.
    return mean, std


class SegmentationImageTransformOnly:
    """
    Apply an image-only transform while leaving the segmentation mask unchanged.

    This is useful for transforms such as:
        - ToTensor
        - Normalize
        - ColorJitter

    These transforms should not be directly applied to masks because masks
    contain discrete class indices.
    """

    def __init__(self, img_transform: Callable) -> None:
        # Store the transform that should be applied only to the image.
        self.img_transform = img_transform

    def __call__(self, input_pair):
        # Unpack the image and mask pair.
        image, mask = input_pair

        # Apply the wrapped transform only to the image.
        image = self.img_transform(image)

        # Return the transformed image and unchanged mask.
        return image, mask


class SegmentationRandomResizedCrop:
    """
    Apply the same random resized crop to both image and mask.

    The image uses bilinear interpolation, while the mask uses nearest-neighbor
    interpolation so class IDs are not corrupted.
    """

    def __init__(
        self,
        resize_size: tuple[int, int] = (352, 640),
        scale: tuple[float, float] = (1.0, 1.0),
        ratio: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        # Store the final output size as (height, width).
        self.resize_size = resize_size

        # Store the random area scale range.
        self.scale = scale

        # Store the random aspect-ratio range.
        self.ratio = ratio

    def __call__(self, input_pair):
        # Unpack the image and mask pair.
        image, mask = input_pair

        # Sample one crop configuration from the image.
        i, j, h, w = v2.RandomResizedCrop.get_params(
            image,
            scale=self.scale,
            ratio=self.ratio,
        )

        # Crop the image using the sampled crop box.
        image = TF.crop(image, i, j, h, w)

        # Crop the mask using the exact same crop box.
        mask = TF.crop(mask, i, j, h, w)

        # Resize the image using bilinear interpolation.
        image = TF.resize(
            image,
            self.resize_size,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        # Add a channel dimension because torchvision resize expects [C, H, W].
        mask = mask.unsqueeze(0)

        # Resize the mask using nearest-neighbor interpolation.
        mask = TF.resize(
            mask,
            self.resize_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # Remove the temporary channel dimension from the mask.
        mask = mask.squeeze(0)

        # Return the transformed pair.
        return image, mask


class SegmentationRandomRotation:
    """
    Apply the same random rotation to both image and mask.

    The mask uses nearest-neighbor interpolation to preserve class IDs.
    """

    def __init__(self, degrees: float = 15) -> None:
        # Store the maximum absolute rotation angle.
        self.degrees = degrees

    def __call__(self, input_pair):
        # Unpack the image and mask pair.
        image, mask = input_pair

        # Sample a random angle uniformly from [-degrees, degrees].
        angle = random.uniform(-self.degrees, self.degrees)

        # Rotate the image.
        image = TF.rotate(image, angle)

        # Add a channel dimension to the mask before rotating.
        mask = mask.unsqueeze(0)

        # Rotate the mask with nearest-neighbor interpolation and background fill.
        mask = TF.rotate(
            mask,
            angle,
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )

        # Remove the temporary channel dimension from the mask.
        mask = mask.squeeze(0)

        # Return the transformed pair.
        return image, mask


class SegmentationRandomFlip:
    """
    Randomly apply a horizontal flip to both image and mask.
    """

    def __init__(self, prob: float = 0.5) -> None:
        # Store the probability of applying the flip.
        self.prob = prob

    def __call__(self, input_pair):
        # Apply the flip only with probability self.prob.
        if random.random() < self.prob:
            # Unpack the image and mask pair.
            image, mask = input_pair

            # Flip the image horizontally.
            image = TF.hflip(image)

            # Add a channel dimension to the mask before flipping.
            mask = mask.unsqueeze(0)

            # Flip the mask horizontally using the same operation.
            mask = TF.hflip(mask)

            # Remove the temporary channel dimension from the mask.
            mask = mask.squeeze(0)

            # Return the flipped pair.
            return image, mask

        # If the random condition is not met, return the pair unchanged.
        return input_pair
