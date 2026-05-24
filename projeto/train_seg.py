"""
train_seg.py

Training entry point for the complete segmentation pipeline.

This script integrates:
    - segmentation_dataset.py
    - transforms.py
    - seg_data_module.py
    - all_backbones.py
    - downstream_segmentation.py

Run example
-----------
python train_seg.py --config config.yaml

Required config.yaml keys
-------------------------
BATCH_SIZE
NUM_EPOCHS
INITIAL_LR
NUM_WORKERS
experiment_group_name
architecture
pretrained
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import lightning as pl
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


from backbones.all_backbones import UniversalBackbone
from downstream.downstream_segmentation import LitAdvancedSegmentationModel
from data_processing.seg_data_module import SegmentationLightningDataModule
from data_processing.segmentation_dataset import (
    SEG_DATASET_URL,
    download_and_extract_segmentation_dataset,
    load_segmentation_dataset_config,
)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    # Convert to pathlib Path.
    config_path = Path(config_path)

    # Raise a clear error if the file is missing.
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read the YAML file.
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Protect against empty YAML files.
    if config is None:
        config = {}

    # Return the parsed config.
    return config


def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get a config value with uppercase/lowercase compatibility.
    """

    # Return the exact key if available.
    if key in config:
        return config[key]

    # Return the lowercase key if available.
    lower_key = key.lower()
    if lower_key in config:
        return config[lower_key]

    # Return the default if neither key is available.
    return default


def str_to_bool(value: Any) -> bool:
    """
    Convert YAML/string values into a reliable boolean.
    """

    # If the value is already boolean, return it directly.
    if isinstance(value, bool):
        return value

    # If the value is numeric, follow normal Python truthiness.
    if isinstance(value, (int, float)):
        return bool(value)

    # Convert strings such as "true", "yes", and "1" to True.
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}

    # Fall back to Python truthiness.
    return bool(value)





def tensor_or_value_to_python(value: Any) -> Any:
    """
    Convert a Lightning/Torch metric value into a CSV-friendly Python value.

    Notes
    -----
    Lightning metrics are often returned as scalar tensors. CSV files cannot
    directly store tensors, so this helper converts:
        - scalar tensors into floats
        - vector tensors into Python lists
        - normal Python values into themselves
    """

    # Avoid importing torch explicitly here; checking for detach keeps this
    # helper compatible with tensor-like metric objects returned by Lightning.
    if hasattr(value, "detach"):
        # Move the tensor to CPU and remove it from the computation graph.
        value = value.detach().cpu()

        # Convert scalar tensors into Python floats.
        if value.numel() == 1:
            return float(value.item())

        # Convert vector tensors into Python lists.
        return value.tolist()

    # Return non-tensor values unchanged.
    return value


def metric_to_float(metrics: Dict[str, Any], key: str) -> float | None:
    """
    Read one scalar metric from a Lightning result dictionary.
    """

    # Return None when the metric key does not exist.
    if key not in metrics:
        return None

    # Convert the metric value to a Python scalar/list if needed.
    value = tensor_or_value_to_python(metrics[key])

    # Scalar metrics should become floats.
    if isinstance(value, (int, float)):
        return float(value)

    # Return None if the value is not scalar.
    return None


def collect_per_class_metrics(
    metrics: Dict[str, Any],
    metric_prefix: str,
    num_classes: int,
) -> List[float | None]:
    """
    Collect per-class metrics logged by downstream_segmentation.py.

    downstream_segmentation.py logs per-class metrics as separate keys, for
    example:
        val_iou_per_class_0
        val_iou_per_class_1
        test_f1_per_class_0
        test_f1_per_class_1

    This function rebuilds those separate keys into one ordered list so it can
    be saved in one CSV cell as JSON text.
    """

    # Create an empty list that will receive class-wise values in class order.
    values: List[float | None] = []

    # Loop over the expected class indices.
    for class_index in range(num_classes):
        # Build the exact key produced by downstream_segmentation.py.
        key = f"{metric_prefix}_{class_index}"

        # Append the scalar value, or None if the key is missing.
        values.append(metric_to_float(metrics, key))

    # Return the ordered per-class metric list.
    return values


def count_training_images(data_module: Any) -> int | None:
    """
    Try to infer the number of training images from the LightningDataModule.

    Different DataModule implementations use different attribute names. This
    function checks common names without requiring changes to seg_data_module.py.
    """

    # Make sure setup has created dataset attributes when possible.
    try:
        data_module.setup("fit")
    except TypeError:
        data_module.setup()
    except Exception:
        # If setup fails here, training may still call setup internally later.
        pass

    # Check common dataset attribute names.
    for attribute_name in ("train_dataset", "dataset_train", "training_dataset"):
        # Read the attribute if it exists.
        dataset = getattr(data_module, attribute_name, None)

        # Return its length if available.
        if dataset is not None and hasattr(dataset, "__len__"):
            return int(len(dataset))

    # Fall back to the train dataloader dataset.
    try:
        train_loader = data_module.train_dataloader()
        return int(len(train_loader.dataset))
    except Exception:
        return None


def append_experiment_row_to_csv(csv_path: str | Path, row: Dict[str, Any]) -> None:
    """
    Append one experiment result row to a CSV file.

    The CSV file is created with a header if it does not already exist.
    List-valued metrics are stored as JSON strings inside single CSV cells.
    """

    # Convert the input path to pathlib.Path.
    csv_path = Path(csv_path)

    # Create the parent directory when the path includes one.
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Define the exact requested CSV column order.
    fieldnames = [
        "experiment_group_name",
        "training_mode",
        "backbone",
        "freeze_backbone",
        "num_epochs",
        "batch _size",
        "image_size",
        "initial_lr",
        "number_of_training_images",
        "val_f1_score",
        "val_iou",
        "val_iou_per_class",
        "val_f1_per_class",
        "test_f1_score",
        "test_iou",
        "test_iou_per_class",
        "test_f1_per_class",
    ]

    # Convert lists/dicts to JSON strings so the CSV stays one row per run.
    csv_row = {
        key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value
        for key, value in row.items()
    }

    # Check whether the CSV file already exists.
    file_already_exists = csv_path.is_file()

    # Append the row to the CSV file.
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        # Create a DictWriter with the fixed column order.
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header only when creating a new CSV file.
        if not file_already_exists:
            writer.writeheader()

        # Write one experiment row.
        writer.writerow(csv_row)


def main() -> None:
    """
    Train the segmentation model from a YAML config file.
    """

    # Create the command-line parser.
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ with a UniversalBackbone."
    )

    # Add the required config argument.
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config.yaml file.",
    )

    # Parse command-line arguments.
    args = parser.parse_args()

    # Load the YAML configuration.
    config = load_yaml_config(args.config)

    # Read core training values.
    batch_size = int(get_config_value(config, "BATCH_SIZE", 16))
    num_epochs = int(get_config_value(config, "NUM_EPOCHS", 50))
    initial_lr = float(get_config_value(config, "INITIAL_LR", 1e-3))
    num_workers = int(get_config_value(config, "NUM_WORKERS", 2))

    # Read experiment naming settings.
    experiment_group_name = str(
        get_config_value(config, "experiment_group_name", "deeplabv3plus_segmentation")
    )

    # Read UniversalBackbone architecture.
    architecture = str(get_config_value(config, "architecture", "resnet50"))

    # Read whether the UniversalBackbone starts from pretrained ImageNet weights.
    pretrained = str_to_bool(get_config_value(config, "pretrained", True))

    # Read optional separate backbone learning rate.
    backbone_lr = get_config_value(config, "BACKBONE_LR", None)
    backbone_lr = None if backbone_lr is None else float(backbone_lr)

    # Read optional freeze-backbone setting.
    freeze_backbone = str_to_bool(get_config_value(config, "freeze_backbone", False))

    # Read optional final image size.
    image_size = get_config_value(config, "IMAGE_SIZE", [352, 640])
    image_size = tuple(int(value) for value in image_size)

    # Read optional dataset and output paths.
    data_folder = Path(get_config_value(config, "DATA_FOLDER", "data"))
    checkpoint_root = Path(get_config_value(config, "CHECKPOINT_ROOT", "my_seg_models"))
    tb_log_root = Path(get_config_value(config, "TB_LOG_ROOT", "tb_logs"))
    csv_results_path = get_config_value(config, "CSV_RESULTS_PATH", "experiment_results.csv")
    csv_results_path = Path(csv_results_path + f"{experiment_group_name}.csv")  # Add group name to CSV for better organization.

    # Read optional dataset download behavior.
    download_dataset = str_to_bool(get_config_value(config, "DOWNLOAD_DATASET", True))

    # Read optional seed.
    seed = get_config_value(config, "SEED", None)

    # Seed Lightning and DataLoader workers if a seed was provided.
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    # Download/extract the dataset or use an existing dataset folder.
    if download_dataset:
        dataset_folder = download_and_extract_segmentation_dataset(
            data_folder=data_folder,
            dataset_url=str(get_config_value(config, "SEG_DATASET_URL", SEG_DATASET_URL)),
        )
    else:
        dataset_folder = Path(
            get_config_value(
                config,
                "SEG_DATASET_FOLDER",
                data_folder / "segmentation_640x360",
            )
        )

    # Load the dataset metadata JSON.
    seg_dataset_config = load_segmentation_dataset_config(dataset_folder)

    # Read the number of foreground segmentation classes.
    num_valid_classes = int(seg_dataset_config["num_classes"])

    # Build a timestamped run name.
    date_str = datetime.now().strftime("%Y-%m-%d-%Hh%M")

    # Include the main experiment settings in the run name.
    run_name = (
        f"arch_{architecture}_pretrained_{pretrained}_"
        f"lr_{initial_lr}_bs_{batch_size}_epochs_{num_epochs}_date_{date_str}"
    )
    print('run name: ', run_name)
    # Create the TensorBoard logger.
    tb_logger = TensorBoardLogger(
        save_dir=str(tb_log_root),
        name=experiment_group_name,
        version=run_name,
    )

    # Create the checkpoint callback.
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_root / experiment_group_name / run_name),
        filename="best-model",
        save_top_k=1,
        mode="min",
    )

    # Create the integrated LightningDataModule.
    seg_data_module = SegmentationLightningDataModule(
        data_dir=dataset_folder,
        num_valid_classes=num_valid_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )

    # Count the number of training images before training.
    number_of_training_images = count_training_images(seg_data_module)

    # Create the notebook-style UniversalBackbone.
    universal_backbone = UniversalBackbone(
        architecture=architecture,
        pretrained=pretrained,
    )
    print(f"UniversalBackbone created with architecture={architecture} and pretrained={pretrained}")
    print(universal_backbone.backbone_model)  # Print the backbone architecture for verification.

    # Create the Lightning segmentation model.
    seg_model = LitAdvancedSegmentationModel(
        num_valid_classes=num_valid_classes,
        universal_backbone=universal_backbone,
        backbone_architecture=architecture,
        backbone_pretrained=pretrained,
        learning_rate=initial_lr,
        backbone_lr=backbone_lr,
        freeze_backbone=freeze_backbone,
    )

    # Create the Lightning trainer.
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=int(get_config_value(config, "LOG_EVERY_N_STEPS", 10)),
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )

    # Train the model.
    trainer.fit(seg_model, datamodule=seg_data_module)

    # Select the best checkpoint if Lightning saved one.
    best_checkpoint_path = checkpoint_callback.best_model_path

    # Use the best checkpoint for final validation/test metrics when available.
    checkpoint_for_metrics = best_checkpoint_path if best_checkpoint_path else None

    # Run validation after training so the final CSV receives stable val metrics.
    validation_results = trainer.validate(
        model=seg_model,
        datamodule=seg_data_module,
        ckpt_path=checkpoint_for_metrics,
        verbose=False,
    )

    # Run test after training so the final CSV receives test metrics.
    test_results = trainer.test(
        model=seg_model,
        datamodule=seg_data_module,
        ckpt_path=checkpoint_for_metrics,
        verbose=False,
    )

    # Lightning returns a list with one metrics dictionary per dataloader.
    val_metrics = validation_results[0] if validation_results else {}
    test_metrics = test_results[0] if test_results else {}

    # Build the CSV row using the metric names produced by downstream_segmentation.py.
    csv_row = {
        "experiment_group_name": experiment_group_name,
        "training_mode": "pretrained" if pretrained else "from scratch",
        "backbone": architecture,
        "freeze_backbone": freeze_backbone,
        "num_epochs": num_epochs,
        "batch _size": batch_size,
        "image_size": list(image_size),
        "initial_lr": initial_lr,
        "number_of_training_images": number_of_training_images,
        "val_f1_score": metric_to_float(val_metrics, "val_f1"),
        "val_iou": metric_to_float(val_metrics, "val_iou"),
        "val_iou_per_class": collect_per_class_metrics(
            val_metrics,
            "val_iou_per_class",
            num_valid_classes,
        ),
        "val_f1_per_class": collect_per_class_metrics(
            val_metrics,
            "val_f1_per_class",
            num_valid_classes,
        ),
        "test_f1_score": metric_to_float(test_metrics, "test_f1"),
        "test_iou": metric_to_float(test_metrics, "test_iou"),
        "test_iou_per_class": collect_per_class_metrics(
            test_metrics,
            "test_iou_per_class",
            num_valid_classes,
        ),
        "test_f1_per_class": collect_per_class_metrics(
            test_metrics,
            "test_f1_per_class",
            num_valid_classes,
        ),
    }

    # Append the experiment metadata and metrics to the configured CSV file.
    append_experiment_row_to_csv(csv_results_path, csv_row)

    # Print the CSV location for verification.
    print(f"Experiment metrics saved to: {csv_results_path}")

    # Print the best checkpoint location.
    print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")

    # Export the trained backbone if requested.
    if str_to_bool(get_config_value(config, "EXPORT_TRAINED_BACKBONE", True)):
        # Extract the trained backbone from the segmentation model.
        trained_backbone = seg_model.get_trained_universal_backbone()

        # Define the destination path.
        backbone_path = (
            checkpoint_root
            / experiment_group_name
            / run_name
            / "trained_universal_backbone.pt"
        )

        # Save the trained backbone checkpoint.
        trained_backbone.save_pretrained(backbone_path)


# Execute main() only when this file is run as a script.
if __name__ == "__main__":
    main()
