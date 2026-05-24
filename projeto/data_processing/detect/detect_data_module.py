import lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os
import json
from data_processing.detect.detect_dataset import DetectionDataset
from data_processing.detect.detect_transforms import DetectionRandomResizedCrop, DetectionImageTransformOnly



def detection_collate_fn(batch):
    """
    Custom collate function for object detection.

    Faster R-CNN expects a list of images and a list of target dictionaries.
    Since each image can have a different number of objects (bounding boxes),
    targets cannot be stacked into a single tensor.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

#pipeline train and validation
class DetectionLightningDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 2, train_transform=None, val_test_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # No need to apply T.Normalize here -- Faster R-CNN performs internal normalization
        # per sample; converting to a tensor is sufficient.
        if train_transform:
            self.train_transform = train_transform
        else:
            self.train_transform = T.Compose([
                DetectionRandomResizedCrop(resize_size=(352,640), scale=(0.5, 1.0), ratio=(3.0/4.0,4.0/3.0)),
                DetectionImageTransformOnly(T.ToTensor())
            ])
        if val_test_transform:
            self.val_test_transform = val_test_transform
        else:
            self.val_test_transform = T.Compose([
                DetectionRandomResizedCrop(resize_size=(352,640), scale=(1.0, 1.0), ratio=(1.0,1.0)),  # Resize only.
                DetectionImageTransformOnly(T.ToTensor())
            ])

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            with open(os.path.join(self.data_dir, 'train.json')) as f:
                train_coco = json.load(f)
            with open(os.path.join(self.data_dir, 'val.json')) as f:
                val_coco = json.load(f)

            self.train_ds = DetectionDataset(
                images_dir=os.path.join(self.data_dir, 'images/train'),
                coco_data=train_coco,
                transform=self.train_transform,
            )
            self.val_ds = DetectionDataset(
                images_dir=os.path.join(self.data_dir, 'images/val'),
                coco_data=val_coco,
                transform=self.val_test_transform,
            )

        if stage == 'test' or stage is None:
            with open(os.path.join(self.data_dir, 'test.json')) as f:
                test_coco = json.load(f)

            self.test_ds = DetectionDataset(
                images_dir=os.path.join(self.data_dir, 'images/test'),
                coco_data=test_coco,
                transform=self.val_test_transform,
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=detection_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=detection_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=detection_collate_fn)
