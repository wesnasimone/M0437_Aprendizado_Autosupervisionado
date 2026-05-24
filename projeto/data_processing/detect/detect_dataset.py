import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import os


class DetectionDataset(Dataset):
    """
    Custom Dataset for object detection in PyTorch.

    Args:
        images_dir (str): Path to the directory containing the images for the split.
        coco_data (dict): Dictionary containing the 'images', 'annotations', and '
                          categories' keys, loaded from the COCO JSON file corresponding
                          to the split.
        transform (callable, optional): Optional transformations to be applied to the
                                        images. If None, the image is converted to a
                                        tensor [0, 1] via ToTensor.
    """
    def __init__(self, images_dir: str, coco_data: dict, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        # Image metadata list (contains 'id', 'file_name', etc.)
        self.images_info = coco_data['images']

        # Fast access index: image_id -> list of annotations
        self.anns_by_image = {}
        for ann in coco_data['annotations']:
            self.anns_by_image.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.images_info)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])

        # 1. Load image and record ORIGINAL dimensions
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # 2. Prepare the target (bounding boxes and labels)
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in self.anns_by_image.get(img_info['id'], []):
            x, y, w, h = ann['bbox']
            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            # as required by Torchvision models.
            x1, y1, x2, y2 = x, y, x+w, y+h

            # Ensure bounding boxes stay within image boundaries (clipping)
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # Discard boxes with zero or negative area after clipping
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])

            # Increment category_id by 1 (Background is ID 0)
            labels.append(ann['category_id'] + 1)
            areas.append(w * h)
            iscrowd.append(ann['iscrowd'])

        target = {
            'boxes':    torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),
            'area':     torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd':  torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        # 3. Apply transformations (e.g., Resize, Normalization, Augmentation)
        if self.transform:
            image, target = self.transform( (image, target) )
            return image, target
        else:
            image = TF.to_tensor(image)
            return image, target
