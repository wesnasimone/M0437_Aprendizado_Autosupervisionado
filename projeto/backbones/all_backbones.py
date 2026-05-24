"""
all_backbones.py

Universal backbone definitions for the segmentation pipeline.

This module keeps the external API used by train_seg.py:

    universal_backbone = UniversalBackbone(
        architecture="vgg11",
        pretrained=True,
    )

or:

    universal_backbone = UniversalBackbone(
        architecture="vit_16",
        pretrained=True,
    )

The important design rule is:

    UniversalBackbone owns only the reusable feature extractor/backbone.
    LitAdvancedSegmentationModel owns the final segmentation model.

Special cases
-------------
VGG11:
    segmentation_models_pytorch.DeepLabV3Plus cannot use VGG encoders because
    SMP's VGG encoder does not support dilated mode. Therefore VGG11 is adapted
    for torchvision.models.segmentation.DeepLabV3.

ViT-B/16:
    Vision Transformers are not CNN encoders and do not naturally expose a
    spatial feature map. Therefore ViT-B/16 is adapted by converting patch tokens
    back into a feature map with shape [B, hidden_dim, H/16, W/16]. This feature
    map is then usable by torchvision DeepLabV3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    DenseNet169_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    VGG11_Weights,
    ViT_B_16_Weights,
    densenet169,
    efficientnet_b0,
    resnet50,
    vgg11,
    vit_b_16,
)


def normalize_architecture_name(architecture: str) -> str:
    """
    Convert common architecture aliases into one canonical name.

    This lets config.yaml use either "vit_16", "vit-b-16", "vit_b_16",
    or "vit_b_16" without changing train_seg.py.
    """

    # Normalize spacing/case first.
    architecture = str(architecture).strip().lower()

    # Accept common aliases for ViT-B/16.
    if architecture in {"vit_16", "vit-b-16", "vit_b_16", "vitb16", "vit-b16"}:
        return "vit_b_16"

    # Keep existing architecture names unchanged.
    return architecture


class VGG11DeepLabV3Backbone(nn.Module):
    """
    VGG11 feature extractor adapted for torchvision DeepLabV3.

    torchvision DeepLabV3 expects the backbone to return a dictionary:

        {"out": feature_tensor}

    and to expose:

        out_channels

    This wrapper satisfies that interface while preserving ImageNet-pretrained
    VGG11 convolutional weights when pretrained=True.
    """

    def __init__(self, pretrained: bool = True) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Select ImageNet pretrained weights only when requested.
        weights = VGG11_Weights.DEFAULT if pretrained else None

        # Build the torchvision VGG11 classification model.
        vgg_model = vgg11(weights=weights)

        # Keep only the convolutional feature extractor.
        self.features = vgg_model.features

        # VGG11's final convolutional feature map has 512 channels.
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return a dictionary because torchvision DeepLabV3 requires it.
        """

        # Compute the final VGG feature map.
        feature_map = self.features(x)

        # DeepLabV3 reads features["out"] internally.
        return {"out": feature_map}


class VGG11FasterRCNN(nn.Module):
    """
    VGG11 feature extractor adapted for torchvision FasterRCNN.

    """

    def __init__(self, pretrained: bool = True) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Select ImageNet pretrained weights only when requested.
        weights = VGG11_Weights.DEFAULT if pretrained else None

        # Build the torchvision VGG11 classification model.
        vgg_model = vgg11(weights=weights)

        # Keep only the convolutional feature extractor.
        self.features = vgg_model.features

        # VGG11's final convolutional feature map has 512 channels.
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Compute the final VGG feature map.
        feature_map = self.features(x)

        return feature_map


class ViT16DeepLabV3Backbone(nn.Module):
    """
    ViT-B/16 feature extractor adapted for torchvision DeepLabV3.

    Why this class is needed
    ------------------------
    A normal ViT returns classification tokens, not CNN-style feature maps.
    DeepLabV3 needs a spatial feature map. This wrapper:

        1. Loads torchvision ViT-B/16, optionally with ImageNet weights.
        2. Runs the patch embedding and transformer encoder.
        3. Drops the class token.
        4. Reshapes patch tokens into [B, 768, H/16, W/16].
        5. Returns {"out": feature_map} for torchvision DeepLabV3.

    Important limitation
    --------------------
    This is not the original DeepLabV3 paper design because ViT-B/16 is not a
    dilated CNN. However, it is a practical way to reuse pretrained ViT weights
    inside the same DeepLabV3 segmentation training pipeline.
    """

    def __init__(self, pretrained: bool = True) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Select ImageNet pretrained weights only when requested.
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None

        # Build torchvision ViT-B/16.
        vit_model = vit_b_16(weights=weights)

        # Keep the patch embedding convolution.
        # For ViT-B/16 this converts [B, 3, H, W] into [B, 768, H/16, W/16].
        self.conv_proj = vit_model.conv_proj

        # Keep the learnable class token.
        self.class_token = vit_model.class_token

        # Keep the transformer encoder.
        self.encoder = vit_model.encoder

        # Keep important metadata.
        self.patch_size = 16
        self.out_channels = int(vit_model.hidden_dim)

        # Store the image size used by the pretrained positional embedding.
        # For torchvision ViT-B/16 ImageNet weights this is normally 224.
        self.pretrained_image_size = int(vit_model.image_size)

    def _interpolate_positional_embedding(
        self,
        num_patch_rows: int,
        num_patch_cols: int,
    ) -> torch.Tensor:
        """
        Resize pretrained ViT positional embeddings to the current image size.

        torchvision ViT-B/16 pretrained weights are learned for 224x224 images,
        giving a 14x14 patch grid. Segmentation images are often larger or
        non-square, so the positional embedding must be interpolated to match
        the runtime patch grid.
        """

        # Read the learned positional embedding from the torchvision encoder.
        # Shape: [1, 1 + old_num_patches, hidden_dim].
        pos_embedding = self.encoder.pos_embedding

        # Separate class-token position from patch-token positions.
        class_pos_embedding = pos_embedding[:, :1, :]
        patch_pos_embedding = pos_embedding[:, 1:, :]

        # Recover the original square patch grid size, usually 14 for 224/16.
        old_num_patches = patch_pos_embedding.shape[1]
        old_grid_size = int(old_num_patches ** 0.5)

        # Convert patch position embeddings from sequence layout to image layout.
        # [1, old_patches, C] -> [1, C, old_grid, old_grid].
        patch_pos_embedding = patch_pos_embedding.reshape(
            1,
            old_grid_size,
            old_grid_size,
            self.out_channels,
        )
        patch_pos_embedding = patch_pos_embedding.permute(0, 3, 1, 2)

        # Resize the positional grid to match the current patch grid.
        patch_pos_embedding = F.interpolate(
            patch_pos_embedding,
            size=(num_patch_rows, num_patch_cols),
            mode="bicubic",
            align_corners=False,
        )

        # Convert back from image layout to sequence layout.
        # [1, C, new_rows, new_cols] -> [1, new_patches, C].
        patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1)
        patch_pos_embedding = patch_pos_embedding.reshape(
            1,
            num_patch_rows * num_patch_cols,
            self.out_channels,
        )

        # Reattach the class-token positional embedding.
        return torch.cat([class_pos_embedding, patch_pos_embedding], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return a ViT patch-token feature map as {"out": feature_map}.
        """

        # ViT-B/16 requires dimensions divisible by 16 because patches are 16x16.
        image_height, image_width = x.shape[-2:]
        if image_height % self.patch_size != 0 or image_width % self.patch_size != 0:
            raise ValueError(
                "ViT-B/16 DeepLabV3 backbone requires image height and width "
                f"to be divisible by {self.patch_size}. "
                f"Got input shape {tuple(x.shape)}."
            )

        # Convert the image to patch embeddings.
        # Shape: [B, 768, H/16, W/16].
        x = self.conv_proj(x)

        # Save the patch grid size before flattening.
        num_patch_rows, num_patch_cols = x.shape[-2:]

        # Convert from image layout to token-sequence layout.
        # [B, C, Hp, Wp] -> [B, Hp*Wp, C].
        x = x.flatten(2).transpose(1, 2)

        # Expand the class token to the batch size.
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)

        # Prepend the class token, matching normal ViT behavior.
        x = torch.cat([batch_class_token, x], dim=1)

        # Add positional embeddings, interpolated for the current patch grid.
        x = x + self._interpolate_positional_embedding(
            num_patch_rows=num_patch_rows,
            num_patch_cols=num_patch_cols,
        ).to(dtype=x.dtype, device=x.device)

        # Run the transformer encoder.
        # We do this manually instead of calling self.encoder(x) because the
        # default encoder expects the original positional embedding size.
        x = self.encoder.dropout(x)
        x = self.encoder.layers(x)
        x = self.encoder.ln(x)

        # Drop the class token because segmentation needs spatial patch tokens.
        patch_tokens = x[:, 1:, :]

        # Convert tokens back into a spatial feature map.
        # [B, Hp*Wp, C] -> [B, C, Hp, Wp].
        feature_map = patch_tokens.transpose(1, 2).reshape(
            x.shape[0],
            self.out_channels,
            num_patch_rows,
            num_patch_cols,
        )

        # DeepLabV3 reads features["out"] internally.
        return {"out": feature_map}


class ViT16FasterRCNN(nn.Module):
    """
    ViT-B/16 feature extractor adapted for torchvision DeepLabV3.

    Why this class is needed
    ------------------------
    A normal ViT returns classification tokens, not CNN-style feature maps.
    DeepLabV3 needs a spatial feature map. This wrapper:

        1. Loads torchvision ViT-B/16, optionally with ImageNet weights.
        2. Runs the patch embedding and transformer encoder.
        3. Drops the class token.
        4. Reshapes patch tokens into [B, 768, H/16, W/16].
        5. Returns {"out": feature_map} for torchvision DeepLabV3.

    Important limitation
    --------------------
    This is not the original DeepLabV3 paper design because ViT-B/16 is not a
    dilated CNN. However, it is a practical way to reuse pretrained ViT weights
    inside the same DeepLabV3 segmentation training pipeline.
    """

    def __init__(self, pretrained: bool = True) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Select ImageNet pretrained weights only when requested.
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None

        # Build torchvision ViT-B/16.
        vit_model = vit_b_16(weights=weights)

        # Keep the patch embedding convolution.
        # For ViT-B/16 this converts [B, 3, H, W] into [B, 768, H/16, W/16].
        self.conv_proj = vit_model.conv_proj

        # Keep the learnable class token.
        self.class_token = vit_model.class_token

        # Keep the transformer encoder.
        self.encoder = vit_model.encoder

        # Keep important metadata.
        self.patch_size = 16
        self.out_channels = int(vit_model.hidden_dim)

        # Store the image size used by the pretrained positional embedding.
        # For torchvision ViT-B/16 ImageNet weights this is normally 224.
        self.pretrained_image_size = int(vit_model.image_size)

    def _interpolate_positional_embedding(
        self,
        num_patch_rows: int,
        num_patch_cols: int,
    ) -> torch.Tensor:
        """
        Resize pretrained ViT positional embeddings to the current image size.

        torchvision ViT-B/16 pretrained weights are learned for 224x224 images,
        giving a 14x14 patch grid. Segmentation images are often larger or
        non-square, so the positional embedding must be interpolated to match
        the runtime patch grid.
        """

        # Read the learned positional embedding from the torchvision encoder.
        # Shape: [1, 1 + old_num_patches, hidden_dim].
        pos_embedding = self.encoder.pos_embedding

        # Separate class-token position from patch-token positions.
        class_pos_embedding = pos_embedding[:, :1, :]
        patch_pos_embedding = pos_embedding[:, 1:, :]

        # Recover the original square patch grid size, usually 14 for 224/16.
        old_num_patches = patch_pos_embedding.shape[1]
        old_grid_size = int(old_num_patches ** 0.5)

        # Convert patch position embeddings from sequence layout to image layout.
        # [1, old_patches, C] -> [1, C, old_grid, old_grid].
        patch_pos_embedding = patch_pos_embedding.reshape(
            1,
            old_grid_size,
            old_grid_size,
            self.out_channels,
        )
        patch_pos_embedding = patch_pos_embedding.permute(0, 3, 1, 2)

        # Resize the positional grid to match the current patch grid.
        patch_pos_embedding = F.interpolate(
            patch_pos_embedding,
            size=(num_patch_rows, num_patch_cols),
            mode="bicubic",
            align_corners=False,
        )

        # Convert back from image layout to sequence layout.
        # [1, C, new_rows, new_cols] -> [1, new_patches, C].
        patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1)
        patch_pos_embedding = patch_pos_embedding.reshape(
            1,
            num_patch_rows * num_patch_cols,
            self.out_channels,
        )

        # Reattach the class-token positional embedding.
        return torch.cat([class_pos_embedding, patch_pos_embedding], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        # ViT-B/16 requires dimensions divisible by 16 because patches are 16x16.
        image_height, image_width = x.shape[-2:]
        if image_height % self.patch_size != 0 or image_width % self.patch_size != 0:
            raise ValueError(
                "ViT-B/16 DeepLabV3 backbone requires image height and width "
                f"to be divisible by {self.patch_size}. "
                f"Got input shape {tuple(x.shape)}."
            )

        # Convert the image to patch embeddings.
        # Shape: [B, 768, H/16, W/16].
        x = self.conv_proj(x)

        # Save the patch grid size before flattening.
        num_patch_rows, num_patch_cols = x.shape[-2:]

        # Convert from image layout to token-sequence layout.
        # [B, C, Hp, Wp] -> [B, Hp*Wp, C].
        x = x.flatten(2).transpose(1, 2)

        # Expand the class token to the batch size.
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)

        # Prepend the class token, matching normal ViT behavior.
        x = torch.cat([batch_class_token, x], dim=1)

        # Add positional embeddings, interpolated for the current patch grid.
        x = x + self._interpolate_positional_embedding(
            num_patch_rows=num_patch_rows,
            num_patch_cols=num_patch_cols,
        ).to(dtype=x.dtype, device=x.device)

        # Run the transformer encoder.
        # We do this manually instead of calling self.encoder(x) because the
        # default encoder expects the original positional embedding size.
        x = self.encoder.dropout(x)
        x = self.encoder.layers(x)
        x = self.encoder.ln(x)

        patch_tokens = x[:, 1:, :]

        # Convert tokens back into a spatial feature map.
        # [B, Hp*Wp, C] -> [B, C, Hp, Wp].
        feature_map = patch_tokens.transpose(1, 2).reshape(
            x.shape[0],
            self.out_channels,
            num_patch_rows,
            num_patch_cols,
        )

        return feature_map


class UniversalBackbone(nn.Module):
    """
    Universal and task-agnostic backbone wrapper.

    Parameters
    ----------
    architecture:
        Backbone name. Supported values are:
            - "resnet50"
            - "efficientnet-b0"
            - "vgg11"
            - "densenet169"
            - "vit_16", "vit_b_16", or "vit-b-16"

    pretrained:
        If True, torchvision ImageNet weights are loaded.
        If False, the architecture is initialized randomly.

    Notes
    -----
    The class stores a reusable backbone and metadata. It does not create the
    final segmentation model. That remains the responsibility of
    LitAdvancedSegmentationModel.
    """

    def __init__(self, architecture: str = "resnet50", pretrained: bool = True) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Store the normalized architecture name.
        self.architecture = normalize_architecture_name(architecture)

        # Store whether pretrained weights were requested.
        self.pretrained = bool(pretrained)

        # The segmentation dataset images are RGB.
        self.in_channels = 3

        # Build the requested backbone.
        if self.architecture == "resnet50":
            # Choose ImageNet weights only when pretrained=True.
            weights = ResNet50_Weights.DEFAULT if pretrained else None

            # Create a torchvision ResNet-50.
            model = resnet50(weights=weights)

            # Expose ResNet stages using names compatible with SMP ResNet encoders.
            self.conv1 = model.conv1
            self.bn1 = model.bn1
            self.relu = model.relu
            self.maxpool = model.maxpool
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4

            # Keep the full original model only as a reference/debug object.
            self.backbone_model = model

            # Store the number of output channels after layer4.
            self.out_channels = 2048

        elif self.architecture == "efficientnet-b0":
            # Choose ImageNet weights only when pretrained=True.
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None

            # Create a torchvision EfficientNet-B0.
            model = efficientnet_b0(weights=weights)

            # Keep only the convolutional feature extractor.
            self.backbone_model = model.features

            # Store the number of output channels from the final feature block.
            self.out_channels = 1280

        elif self.architecture == "vgg11":
            # Build the VGG11 feature extractor adapted for torchvision DeepLabV3.
            self.backbone_model = VGG11DeepLabV3Backbone(pretrained=pretrained)

            # Store the number of output channels expected by DeepLabV3.
            self.out_channels = self.backbone_model.out_channels
        
        elif self.architecture == "vgg11_detect":
            # Build the VGG11 feature extractor adapted for torchvision DeepLabV3.
            self.backbone_model = VGG11FasterRCNN(pretrained=pretrained)

            # Store the number of output channels expected by DeepLabV3.
            self.out_channels = self.backbone_model.out_channels

        elif self.architecture == "vit_b_16":
            # Build the ViT-B/16 feature extractor adapted for torchvision DeepLabV3.
            self.backbone_model = ViT16DeepLabV3Backbone(pretrained=pretrained)

            # Store the number of output channels expected by DeepLabV3.
            self.out_channels = self.backbone_model.out_channels

        elif self.architecture == "vit_b_16_detect":
            # Build the ViT-B/16 feature extractor adapted for torchvision DeepLabV3.
            self.backbone_model = ViT16FasterRCNN(pretrained=pretrained)

            # Store the number of output channels expected by DeepLabV3.
            self.out_channels = self.backbone_model.out_channels

        elif self.architecture == "densenet169":
            # Choose ImageNet weights only when pretrained=True.
            weights = DenseNet169_Weights.DEFAULT if pretrained else None

            # Create a torchvision DenseNet-169.
            model = densenet169(weights=weights)

            # Keep only the convolutional feature extractor.
            self.backbone_model = model.features

            # Store the number of output channels from the final dense block.
            self.out_channels = 1664

        else:
            # Fail clearly for unsupported architectures.
            raise NotImplementedError(
                f"Architecture {architecture!r} is not implemented. "
                "Use one of: resnet50, efficientnet-b0, vgg11, "
                "densenet169, vit_16, vit_b_16, vit-b-16."
            )

    def forward(self, x: torch.Tensor):
        """
        Run the backbone and return its feature representation.

        ResNet returns a plain tensor because it is used mainly for SMP weight
        transfer. VGG11 and ViT-B/16 return {"out": tensor} because torchvision
        DeepLabV3 expects that exact backbone output format.
        """

        # ResNet-50 has named stages, so run them explicitly.
        if self.architecture == "resnet50":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

        # Non-ResNet backbones are stored in backbone_model.
        return self.backbone_model(x)

    def get_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return only the reusable feature-extractor weights.

        This helper is useful because different segmentation wrappers expose
        their backbone under different attributes:

            - SMP models use model.encoder
            - torchvision DeepLabV3 uses model.backbone
        """

        # For ResNet, UniversalBackbone itself contains the named layers.
        if self.architecture == "resnet50":
            return self.state_dict()

        # For the other architectures, the reusable extractor is backbone_model.
        return self.backbone_model.state_dict()

    def save_pretrained(self, filepath: str | Path) -> None:
        """
        Save architecture metadata and backbone weights.
        """

        # Convert the file path into a pathlib Path.
        filepath = Path(filepath)

        # Create the parent directory if needed.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Store enough information to rebuild this backbone.
        state: Dict[str, Any] = {
            "architecture": self.architecture,
            "pretrained": self.pretrained,
            "state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
        }

        # Save the dictionary with torch.save.
        torch.save(state, filepath)

        # Print a friendly confirmation.
        print(f"UniversalBackbone saved at: {filepath}")

    @classmethod
    def load_pretrained(
        cls,
        filepath: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "UniversalBackbone":
        """
        Load a UniversalBackbone saved with save_pretrained().
        """

        # Convert the file path into a pathlib Path.
        filepath = Path(filepath)

        # Load the checkpoint dictionary.
        state = torch.load(filepath, map_location=map_location)

        # Rebuild the architecture without downloading pretrained weights.
        backbone = cls(
            architecture=state["architecture"],
            pretrained=False,
        )

        # Load the saved weights.
        backbone.load_state_dict(state["state_dict"], strict=False)

        # Restore metadata.
        backbone.pretrained = bool(state.get("pretrained", False))
        backbone.in_channels = int(state.get("in_channels", 3))
        backbone.out_channels = int(state.get("out_channels", backbone.out_channels))

        # Print a friendly confirmation.
        print(f"UniversalBackbone loaded from: {filepath}")

        # Return the rebuilt object.
        return backbone
