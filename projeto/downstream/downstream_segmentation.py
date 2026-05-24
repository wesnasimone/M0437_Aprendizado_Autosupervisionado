"""
downstream_segmentation.py

Lightning segmentation model using UniversalBackbone.

This file defines:
    - TorchvisionDeepLabV3FromBackbone
    - LitAdvancedSegmentationModel

Design summary
--------------
- resnet50, efficientnet-b0, and densenet169 use SMP DeepLabV3Plus.
- vgg11 uses torchvision DeepLabV3 because SMP DeepLabV3Plus does not support
  VGG dilation.
- vit_b_16 / vit_16 uses torchvision DeepLabV3 because ViT is not an SMP
  DeepLabV3Plus CNN encoder. The ViT patch tokens are reshaped into a spatial
  feature map by UniversalBackbone.
- train_seg.py does not need to change. It can still create UniversalBackbone
  from config.yaml and pass it into LitAdvancedSegmentationModel.
"""

from __future__ import annotations

from typing import Iterable, Optional

import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelF1Score, MultilabelJaccardIndex
from torchvision.models.segmentation import DeepLabV3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from backbones.all_backbones import UniversalBackbone, normalize_architecture_name


class TorchvisionDeepLabV3FromBackbone(nn.Module):
    """
    torchvision DeepLabV3 using a custom UniversalBackbone-compatible backbone.

    Parameters
    ----------
    backbone:
        A backbone object that returns {"out": feature_map}. In this pipeline it
        is either:

            - VGG11DeepLabV3Backbone
            - ViT16DeepLabV3Backbone

    num_classes:
        Number of output segmentation channels.

    Notes
    -----
    torchvision segmentation models normally return a dictionary:

        {"out": logits}

    This wrapper returns only logits so the rest of the Lightning code can stay
    identical to the SMP path.
    """

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        # Initialize torch.nn.Module internals.
        super().__init__()

        # Store the provided backbone directly.
        # This preserves pretrained weights already loaded by UniversalBackbone.
        self.backbone = backbone

        # Validate that the backbone exposes the channel count needed by DeepLabHead.
        if not hasattr(self.backbone, "out_channels"):
            raise AttributeError(
                "Custom torchvision DeepLabV3 backbones must define "
                "`out_channels`."
            )

        # Build the classifier head used by torchvision DeepLabV3.
        classifier = DeepLabHead(
            in_channels=int(self.backbone.out_channels),
            num_classes=int(num_classes),
        )

        # Build the full torchvision DeepLabV3 model.
        self.deeplab = DeepLabV3(
            backbone=self.backbone,
            classifier=classifier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return segmentation logits with shape [B, num_classes, H, W].
        """

        # torchvision DeepLabV3 returns a dictionary, so unwrap the logits.
        return self.deeplab(x)["out"]

    @property
    def classifier(self) -> nn.Module:
        """
        Expose the segmentation head for optimizer grouping.
        """

        # Return the classifier module inside the torchvision DeepLabV3 model.
        return self.deeplab.classifier


# Backward-compatible alias. Existing imports using the old name still work.
VGG11TorchvisionDeepLabV3 = TorchvisionDeepLabV3FromBackbone


class LitAdvancedSegmentationModel(pl.LightningModule):
    """
    PyTorch Lightning module for semantic segmentation.
    """

    def __init__(
        self,
        num_valid_classes: int,
        universal_backbone,
        backbone_architecture: str = "resnet50",
        backbone_pretrained: bool = True,
        learning_rate: float = 1e-3,
        backbone_lr: Optional[float] = None,
        freeze_backbone: bool = False,
    ) -> None:
        # Initialize LightningModule internals.
        super().__init__()

        # Use the global learning rate for the backbone unless a separate one is given.
        if backbone_lr is None:
            backbone_lr = learning_rate

        # Save simple hyperparameters, but not the full backbone object.
        self.save_hyperparameters(ignore=["universal_backbone"])

        # Store the number of output segmentation channels.
        self.num_valid_classes = int(num_valid_classes)

        # Store backbone metadata from the actual UniversalBackbone object.
        self.backbone_arch = normalize_architecture_name(universal_backbone.architecture)
        self.backbone_pretrained = bool(backbone_pretrained)

        # These architectures need the custom torchvision DeepLabV3 path.
        self.uses_torchvision_deeplab = self.backbone_arch in {"vgg11", "vit_b_16"}

        # ------------------------------------------------------------------
        # Custom-backbone path: torchvision DeepLabV3.
        # ------------------------------------------------------------------
        if self.uses_torchvision_deeplab:
            # UniversalBackbone already contains pretrained VGG11 or ViT weights.
            # We pass that exact object into torchvision DeepLabV3, so no extra
            # load_state_dict call is needed.
            self.model = TorchvisionDeepLabV3FromBackbone(
                backbone=universal_backbone.backbone_model,
                num_classes=num_valid_classes,
            )

            # No load_state_dict is necessary here because the exact pretrained
            # backbone object is already inside self.model.
            self.backbone_load_result = None

        # ------------------------------------------------------------------
        # Standard CNN path: SMP DeepLabV3Plus.
        # ------------------------------------------------------------------
        else:
            # Create a DeepLabV3+ model using SMP's encoder registry.
            # encoder_weights=None prevents SMP from loading its own weights.
            # We inject UniversalBackbone weights manually afterward.
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.backbone_arch,
                in_channels=universal_backbone.in_channels,
                classes=num_valid_classes,
                encoder_weights=None,
            )

            # Inject the UniversalBackbone weights into the SMP encoder.
            load_result = self.model.encoder.load_state_dict(
                universal_backbone.state_dict(),
                strict=False,
            )

            # Keep the load report available for debugging.
            self.backbone_load_result = load_result

        # Print a compact integration report.
        print(f"Backbone architecture: {self.backbone_arch}")
        print(
            "Segmentation backend: "
            + ("torchvision DeepLabV3" if self.uses_torchvision_deeplab else "SMP DeepLabV3Plus")
        )

        # Optionally freeze the encoder/backbone.
        if freeze_backbone:
            for parameter in self._backbone_parameters():
                parameter.requires_grad = False
            print("Segmentation backbone: FROZEN")
        else:
            print("Segmentation backbone: UNFROZEN")

        # Use BCEWithLogitsLoss because masks are one-hot encoded channels.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Define global validation/test metrics.
        global_metrics = MetricCollection(
            {
                "iou": MultilabelJaccardIndex(
                    num_labels=self.num_valid_classes,
                    average="macro",
                ),
                "f1": MultilabelF1Score(
                    num_labels=self.num_valid_classes,
                    average="macro",
                ),
            }
        )

        # Define per-class validation/test metrics.
        per_class_metrics = MetricCollection(
            {
                "iou_per_class": MultilabelJaccardIndex(
                    num_labels=self.num_valid_classes,
                    average="none",
                ),
                "f1_per_class": MultilabelF1Score(
                    num_labels=self.num_valid_classes,
                    average="none",
                ),
            }
        )

        # Clone metric collections so validation and test states do not mix.
        self.val_metrics = global_metrics.clone(prefix="val_")
        self.test_metrics = global_metrics.clone(prefix="test_")
        self.val_metrics_per_class = per_class_metrics.clone(prefix="val_")
        self.test_metrics_per_class = per_class_metrics.clone(prefix="test_")

    def _backbone_parameters(self) -> Iterable[nn.Parameter]:
        """
        Return parameters belonging to the backbone/encoder only.
        """

        # torchvision DeepLabV3 stores the custom backbone as self.model.backbone.
        if self.uses_torchvision_deeplab:
            return self.model.backbone.parameters()

        # SMP DeepLabV3Plus stores the backbone as self.model.encoder.
        return self.model.encoder.parameters()

    def _head_parameters(self) -> Iterable[nn.Parameter]:
        """
        Return parameters belonging to the decoder/classifier head.
        """

        # torchvision DeepLabV3 has a classifier head and no SMP decoder.
        if self.uses_torchvision_deeplab:
            return self.model.classifier.parameters()

        # SMP DeepLabV3Plus has both decoder and segmentation_head modules.
        return list(self.model.decoder.parameters()) + list(self.model.segmentation_head.parameters())

    def get_trained_universal_backbone(self) -> UniversalBackbone:
        """
        Export the trained segmentation backbone back to a UniversalBackbone.
        """

        # Create a clean backbone with the same architecture.
        clean_backbone = UniversalBackbone(
            architecture=self.backbone_arch,
            pretrained=False,
        )

        # Choose the correct trained backbone state depending on the backend.
        if self.uses_torchvision_deeplab:
            trained_state = self.model.backbone.state_dict()
            clean_backbone.backbone_model.load_state_dict(trained_state, strict=False)
        else:
            trained_state = self.model.encoder.state_dict()
            clean_backbone.load_state_dict(trained_state, strict=False)

        # Print a clear confirmation.
        print("UniversalBackbone exported with trained segmentation backbone weights.")

        # Return the exported backbone.
        return clean_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Both backends return logits with shape [B, C, H, W].
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Run one training step.
        """

        # Unpack the image batch and target mask batch.
        images, targets = batch

        # Compute raw logits.
        logits = self(images)

        # Compute the segmentation loss.
        loss = self.loss_fn(logits, targets.float())

        # Log the training loss.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Return the loss for backpropagation.
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Run one validation step.
        """

        # Unpack the image batch and target mask batch.
        images, targets = batch

        # Compute raw logits.
        logits = self(images)

        # Compute validation loss.
        loss = self.loss_fn(logits, targets.float())

        # Convert logits into probabilities for threshold-based multilabel metrics.
        probs = torch.sigmoid(logits)

        # Compute global validation metrics.
        global_metrics = self.val_metrics(probs, targets.long())

        # Log global validation metrics.
        self.log_dict(global_metrics, on_epoch=True, prog_bar=True)

        # Log validation loss.
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        # Compute per-class validation metrics.
        class_metrics = self.val_metrics_per_class(probs, targets.long())

        # Log each per-class metric individually.
        for metric_name, values in class_metrics.items():
            for class_index, value in enumerate(values):
                self.log(f"{metric_name}_{class_index}", value, on_epoch=True)

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Run one test step.
        """

        # Unpack the image batch and target mask batch.
        images, targets = batch

        # Compute raw logits.
        logits = self(images)

        # Compute test loss.
        loss = self.loss_fn(logits, targets.float())

        # Convert logits into probabilities for metrics.
        probs = torch.sigmoid(logits)

        # Log test loss.
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        # Log global test metrics.
        self.log_dict(self.test_metrics(probs, targets.long()), on_epoch=True)

        # Compute per-class test metrics.
        class_metrics = self.test_metrics_per_class(probs, targets.long())

        # Log each per-class metric individually.
        for metric_name, values in class_metrics.items():
            for class_index, value in enumerate(values):
                self.log(f"{metric_name}_{class_index}", value, on_epoch=True)

    def configure_optimizers(self):
        """
        Configure Adam with separate head/backbone parameter groups.
        """

        # Collect trainable decoder/classifier/head parameters.
        head_params = [
            parameter
            for parameter in self._head_parameters()
            if parameter.requires_grad
        ]

        # Collect trainable encoder/backbone parameters.
        backbone_params = [
            parameter
            for parameter in self._backbone_parameters()
            if parameter.requires_grad
        ]

        # Start with an empty list of optimizer groups.
        optimizer_grouped_parameters = []

        # Add decoder/head parameters if they are trainable.
        if head_params:
            optimizer_grouped_parameters.append(
                {
                    "params": head_params,
                    "lr": self.hparams.learning_rate,
                }
            )

        # Add backbone parameters if they are trainable.
        if backbone_params:
            optimizer_grouped_parameters.append(
                {
                    "params": backbone_params,
                    "lr": self.hparams.backbone_lr,
                }
            )

        # Fail clearly if no parameters can be optimized.
        if not optimizer_grouped_parameters:
            raise ValueError("No trainable parameters found. Is the entire model frozen?")

        # Return Adam over the configured parameter groups.
        return torch.optim.Adam(optimizer_grouped_parameters)
