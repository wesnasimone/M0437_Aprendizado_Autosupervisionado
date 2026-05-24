"""
train_detect.py

Training entry point for the complete segmentation pipeline.

This script integrates:
    - detect_dataset.py
    - detect_transforms.py
    - detec_data_module.py
    - all_backbones.py
    - downstream_detection.py
    - plot_prediction.py

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

import os
import json
import urllib.request
import zipfile
from data_processing.detect.detect_dataset import DetectionDataset
from data_processing.detect.detect_transforms import compute_mean_std, DetectionImageTransformOnly, DetectionRandomResizedCrop
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger # 1. Import the logger
from datetime import datetime
import lightning as pl
import torchvision.transforms as T
from data_processing.detect.detect_data_module import DetectionLightningDataModule
from downstream.downstream_detection import LitObjectDetectionModel
from backbones.all_backbones import UniversalBackbone
from pathlib import Path
from typing import Any, Dict, List
import yaml
import argparse
import csv
import wandb
from visualization.detect.plot_prediction import predict_targets, prediction
import random



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
        "initial_lr",
        "number_of_training_images",
        "test_loss",
        "test_mAP",
        "test_mAP_50",
        "test_mAP_class_1",
        "test_mAP_class_2",
        "test_mAP_class_3",
        "test_mAP_class_4"
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


class ToolsWandb:
    @staticmethod
    def config_flatten(config, parent_key='', sep='_'):
        items = []
        for key, value in config.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(ToolsWandb.config_flatten(value, new_key, sep).items())
            else:
                items.append((new_key, value))
        return dict(items)

#MAIN
if __name__ == "__main__":

################## STEP 1 #####################

################# Parameters ##################

    # Create the command-line parser.
    parser = argparse.ArgumentParser(description="Train Detection Task")

    # Add the required config argument.
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file.")

    # Parse command-line arguments.
    args = parser.parse_args()

    # Load the YAML configuration.
    config = load_yaml_config(args.config)


    # Read core training values.
    batch_sizes = int(config["BATCH_SIZE"])
    num_epochs = int(config["NUM_EPOCHS"])
    initial_lr = float(config["INITIAL_LR"])
    num_workers = int(config["NUM_WORKERS"])
    experiment_group_name = str(config["experiment_group_name"])
    freeze_backbone = str_to_bool(config["freeze_backbone"])
    checkpoint_root = Path(config["CHECKPOINT_ROOT"])
    log_every_n_steps = int(config["LOG_EVERY_N_STEPS"])
    path_predict = Path(config["PATH_PREDICT"])

    # Read UniversalBackbone architecture.
    architecture = config["architecture"]

    # Read whether the UniversalBackbone starts from pretrained ImageNet weights.
    pretrained_weights = str_to_bool(config["pretrained"])
    pretrained_str = config["pretrained"]

    #csv path
    csv_results_path = config["CSV_RESULTS_PATH"]
    csv_results_path = Path(csv_results_path + f"{experiment_group_name}.csv")  # Add group name to CSV for better


                                                             
########################## STEP 2 #############################
 
########## Dataset - COCO (Common Objects in Context) #########      
                                                          
    #load dataset or download if it doesn't exist
    DATA_FOLDER         = os.path.join(os.getcwd(), 'data')
    DET_DATASET_ZIPFILE = os.path.join(DATA_FOLDER, 'detection_640x360.zip')
    DET_DATASET_FOLDER  = os.path.join(DATA_FOLDER, 'detection_640x360')

    TARGET_PATH = DET_DATASET_ZIPFILE
    DOWNLOAD_PATH = "https://www.ic.unicamp.br/~edson/disciplinas/mo437/2026-1s/course-work-dataset/detection_640x360.zip"

    # Ensure the data dir exists
    if not os.path.isdir(DATA_FOLDER):
        print(f"Creating the {DATA_FOLDER} folder")
        os.mkdir(DATA_FOLDER)

    # Download if not present
    if not os.path.isfile(DET_DATASET_ZIPFILE):
        print(f"Downloading {TARGET_PATH} from {DOWNLOAD_PATH}")
        urllib.request.urlretrieve(DOWNLOAD_PATH, DET_DATASET_ZIPFILE)

    # Unzip if not unziped
    if not os.path.isdir(DET_DATASET_FOLDER):
        print(f"Unziping dataset into {DET_DATASET_FOLDER} folder")
        with zipfile.ZipFile(TARGET_PATH, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith("detection_640x360"):
                    zip_ref.extract(file, DATA_FOLDER)

    #load dataset config and maps the numerical class IDs to their corresponding human-readable labels.
    det_dataset_config = json.load(open(os.path.join(DET_DATASET_FOLDER, 'dataset_info.json')))

    DET_CATEGORIES = {int(k)+1: v for k, v in det_dataset_config['class_map'].items()}
    DET_CATEGORIES[0] = "background"
    DET_NUM_CLASSES = det_dataset_config['num_classes'] + 1


    #calculate mean and std to each partition in the dataset
    dataset_statistics = {}
    for partition in ["train", "val", "test"]:
        images_dir=os.path.join(DET_DATASET_FOLDER, f"images/{partition}")
        coco_filename = os.path.join(DET_DATASET_FOLDER, f"{partition}.json")
        with open(coco_filename) as f:
            coco_data = json.load(f)
        dataset = DetectionDataset(images_dir=images_dir, coco_data=coco_data)
        mean, std = compute_mean_std(dataset)
        dataset_statistics[partition] = { "mean": mean, "std": std}
    
######################### Settings ############################
    
    # Read optional seed.
    seed = config["SEED"]

    # Seed Lightning and DataLoader workers if a seed was provided.
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)


    # Define the experiment name -- for logging purposes
    formatted_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"arch_{architecture}_pretrained_{pretrained_str}_lr_{initial_lr}_bs_{batch_sizes}_ep_{num_epochs}_{experiment_group_name}-{formatted_date}"

    # Create a tensorboardlogger to log training stats
    tb_logger = TensorBoardLogger(
        save_dir="tb_logs/",
        name=experiment_group_name,
        version=run_name
    )

    #setup wandb
    f_configurations = ToolsWandb.config_flatten(config)

    if config['wandb']["activate"]:
        wandb_run = wandb.init(project=config['wandb']["project"],
                               reinit=True,
                               config=f_configurations,
                               entity=config['wandb']["entity"],
                               save_code=False,
                               name=run_name)
    else:
        wandb_run = None


############################ STEP 3 ##########################

####################### Dataloader + Model ###################

    # Create the notebook-style UniversalBackbone.
    universal_backbone = UniversalBackbone(architecture=architecture,pretrained=pretrained_weights)
    print(f"UniversalBackbone created with architecture={architecture} and pretrained={pretrained_weights}")


    # Create a callback object to save the best checkpoint (minimum validation loss)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{checkpoint_root}/{experiment_group_name}/{run_name}/",
        filename="best-model",
        save_top_k=1,
        mode="min",
    )

    # Instantiate the trainer
    det_trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )

    # Define the augmentation transforms to use during training
    train_transforms = T.Compose([
            DetectionRandomResizedCrop(resize_size=(352,640), scale=(0.5, 1.0), ratio=(0.75,1.25)),
            DetectionImageTransformOnly(T.ToTensor())])

    # Instantiate the data module
    det_data_module = DetectionLightningDataModule(
        data_dir=DET_DATASET_FOLDER,
        train_transform=train_transforms,
        batch_size=batch_sizes,
        num_workers=num_workers,
    )

    # Instantiate the detection model
    det_model = LitObjectDetectionModel(
        num_classes=DET_NUM_CLASSES,
        universal_backbone=universal_backbone,
        learning_rate=initial_lr,
        freeze_backbone=freeze_backbone
    )

    # Count the number of training images before training.
    number_of_training_images = count_training_images(det_data_module)


############################ STEP 4 ##########################

####################### Train ################################

    det_trainer.fit(det_model, datamodule=det_data_module)

######################## Test ################################
    
    best_model = LitObjectDetectionModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        num_classes=DET_NUM_CLASSES,
        universal_backbone=universal_backbone,
        learning_rate=initial_lr,
        freeze_backbone=freeze_backbone
)

    results = det_trainer.test(best_model, datamodule=det_data_module)


    # Prepare the datamodule to for test retrieve the test dataset
    det_data_module.setup(stage="test")
    test_dataset = det_data_module.test_ds

    # Select N random samples
    N = 6
    random_indices = random.sample(range(len(test_dataset)), N)
    samples = [test_dataset[i] for i in random_indices]

    # Run the model
    images, ground_truths, predictions = predict_targets(det_model, samples)
    prediction(images, ground_truths, predictions, path_predict, DET_CATEGORIES, config['wandb']["activate"], N)

    # Build the CSV row using the metric names produced by downstream of detection
    csv_row = {
        "experiment_group_name": experiment_group_name,
        "training_mode": "pretrained" if pretrained_weights else "from scratch",
        "backbone": architecture,
        "freeze_backbone": freeze_backbone,
        "num_epochs": num_epochs,
        "batch _size": batch_sizes,
        "initial_lr": initial_lr,
        "number_of_training_images": number_of_training_images,
        "test_loss": results[0]["test_loss"],
        "test_mAP": results[0]["test_mAP"],
        "test_mAP_50": results[0]["test_mAP_50"],
        #"test_mAP_class_1": results[0]["test_mAP_class_1"],
        #"test_mAP_class_2": results[0]["test_mAP_class_2"],
        #"test_mAP_class_3": results[0]["test_mAP_class_3"],
        #"test_mAP_class_4": results[0]["test_mAP_class_4"]
    }

    # Append the experiment metadata and metrics to the configured CSV file.
    append_experiment_row_to_csv(csv_results_path, csv_row)

    # Print the CSV location for verification.
    print(f"Experiment metrics saved to: {csv_results_path}")

    # Print the best checkpoint location.
    print(f"Best checkpoint path: {checkpoint_callback.best_model_path}")

    # Export the trained backbone if requested.
    if str_to_bool(config["EXPORT_TRAINED_BACKBONE"]):
        # Extract the trained backbone from the segmentation model.
        trained_backbone = det_model.get_trained_universal_backbone()

        # Define the destination path.
        backbone_path = (checkpoint_root/experiment_group_name/run_name/"trained_universal_backbone.pt")

        # Save the trained backbone checkpoint.
        trained_backbone.save_pretrained(backbone_path)
    
    wandb.finish()
