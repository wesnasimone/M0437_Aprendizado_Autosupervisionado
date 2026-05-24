import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from backbones.all_backbones import UniversalBackbone
import lightning as pl
import wandb

class _DetectionBackboneAdapter(torch.nn.Module):
    """
    Private Class: Adapts the UniversalBackbone (which returns a Tensor) to the
    format required by torchvision's Faster R-CNN (which requires an OrderedDict).    """
    def __init__(self, universal_backbone):
        super().__init__()
        self.backbone = universal_backbone
        # torchvision requirement: the wrapper must also pass along
        # the out_channels property
        self.out_channels = universal_backbone.out_channels

    def forward(self, x):
        features = self.backbone(x)
        # MultiScaleRoIAlign is configured to look for the '0' key
        return OrderedDict([('0', features)])


class LitObjectDetectionModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        universal_backbone=UniversalBackbone(),
        learning_rate=1e-3,
        backbone_lr=None,
        freeze_backbone=False
    ):
        super().__init__()

        if backbone_lr is None:
            backbone_lr = learning_rate

        self.save_hyperparameters(ignore=['universal_backbone'])
        self.learning_rate = learning_rate

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        # ADAPTATION: We isolate the torchvision requirement using our local adapter
        adapted_backbone = _DetectionBackboneAdapter(universal_backbone)

        self.model = FasterRCNN(
            adapted_backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            print("Detection Backbone: FROZEN")
        else:
            print("Detection Backbone: UNFREEZED")

        self.val_map = MeanAveragePrecision(class_metrics=True)
        self.test_map = MeanAveragePrecision(class_metrics=True)

    def get_trained_universal_backbone(self):
        """
        Direct access to the backbone trained in detection for unified export.
        """

        # self.model.backbone is our _DetectionBackboneAdapter.
        # Inside it, .backbone is the pure UniversalBackbone instance.
        trained_universal = self.model.backbone.backbone
        print("UniversalBackbone exportado com sucesso com pesos treinados da Detecção.")
        return trained_universal

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        self.log("train_loss", losses, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train_loss": losses})
        for loss_name, loss_val in loss_dict.items():
            self.log(f"train_{loss_name}", loss_val, on_step=True, on_epoch=False)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        was_training = self.model.training

        # --- STEP 1: CALCULATE LOSS ---
        self.model.train()
        with torch.no_grad():
            loss_dict = self(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.log("val_loss", losses, on_epoch=True, prog_bar=True)
            wandb.log({"val_loss": losses})
            

        # --- STEP 2: GET PREDICTIONS FOR mAP ---
        self.model.eval()
        with torch.no_grad():
            # In eval mode, we pass only the images and receive predicted boxes
            preds = self.model(images)

        # Feed the mAP calculator with predictions and ground truths
        self.val_map.update(preds, targets)

        # Restore original state
        self.model.train(mode=was_training)

        return losses

    def on_validation_epoch_end(self):
        """
        Executed at the end of each validation epoch.
        Computes the final aggregated mAP and logs the global and per-class results.
        """
        # compute() returns a dictionary with various metrics (overall mAP, mAP50, mAP75, etc.)
        metrics = self.val_map.compute()

        # Log the most important global results
        self.log("val_mAP", metrics["map"], prog_bar=True)
        self.log("val_mAP_50", metrics["map_50"]) # mAP with IoU > 0.50 (The classic standard)
        wandb.log({"val_mAP": metrics["map"]})
        wandb.log({"val_mAP_50": metrics["map_50"]})

        # Log individual mAP for each class
        if "map_per_class" in metrics:
            map_per_class = metrics["map_per_class"]
            classes = metrics["classes"]

            for cls_idx, map_val in zip(classes, map_per_class):
                # Extract the class integer and log
                class_id = int(cls_idx.item())
                self.log(f"val_mAP_class_{class_id}", map_val)
                wandb.log({f"val_mAP_class_{class_id}": map_val})

        # It is vital to reset the metric at the end of the epoch to avoid leaking data into the next
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        was_training = self.model.training

        # --- STEP 1: CALCULATE LOSS ---
        self.model.train()
        with torch.no_grad():
            loss_dict = self(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            self.log("test_loss", losses, on_epoch=True, prog_bar=True)

        # --- STEP 2: GET PREDICTIONS FOR mAP ---
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images)

        self.test_map.update(preds, targets)
        self.model.train(mode=was_training)

        return losses

    def on_test_epoch_end(self):
        """
        Identical to on_validation_epoch_end, but for the test dataset.
        """
        metrics = self.test_map.compute()

        self.log("test_mAP", metrics["map"])
        self.log("test_mAP_50", metrics["map_50"])

        if "map_per_class" in metrics:
            map_per_class = metrics["map_per_class"]
            classes = metrics["classes"]

            for cls_idx, map_val in zip(classes, map_per_class):
                class_id = int(cls_idx.item())
                self.log(f"test_mAP_class_{class_id}", map_val)

        self.test_map.reset()

    def configure_optimizers(self):

        # 1. Store unique IDs for all parameters belonging to the backbone
        # We use id() as it is the most foolproof way to track tensors in PyTorch for Faster R-CNN networks
        backbone_params_ids = set(id(p) for p in self.model.backbone.parameters())

        backbone_params = []
        other_params = []

        # 2. Iterate through ALL Faster R-CNN parameters
        for p in self.model.parameters():
            # If the parameter was frozen in __init__, we ignore it entirely
            if not p.requires_grad:
                continue

            # If the parameter ID is in our backbone set, it goes to the backbone group
            if id(p) in backbone_params_ids:
                backbone_params.append(p)
            # Otherwise, it belongs to RPN, RoI Heads, etc., and goes to the "other" group
            else:
                other_params.append(p)

        optimizer_grouped_parameters = []

        # 3. Add the main group (RPN and Box/Class predictors) with the default LR
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": self.hparams.learning_rate
            })

        # 4. Add the backbone group with the reduced LR (IF it is not frozen)
        if backbone_params:
            optimizer_grouped_parameters.append({
                "params": backbone_params,
                "lr": self.hparams.backbone_lr
            })

        # Error prevention in case you try to train a 100% frozen network
        if not optimizer_grouped_parameters:
            raise ValueError("Nenhum parâmetro treinável encontrado. O modelo está totalmente congelado?")

        # 5. Instantiate SGD (the gold standard for Faster R-CNN) with the created groups
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            momentum=0.9,
            weight_decay=0.0005
        )

        return optimizer
