import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2
import random
from PIL import Image


def compute_mean_std(img_dataset):
    # Compute the mean
    mean = 0.0
    count = 0
    for image, label in img_dataset:
        flatten_img = torch.flatten(image, start_dim=1)
        mean  += flatten_img.sum(1)  # Sum all the pixes in each channel
        count += flatten_img.size(1) # Count the number of pixes per channel
    mean = mean / count
    # Compute the std
    std = 0.0
    for image, label in img_dataset:
        flatten_img = torch.flatten(image, start_dim=1)
        flatten_img = flatten_img.permute(1,0) - mean
        flatten_img = torch.pow(flatten_img,2)
        std  += flatten_img.sum(0)  # Sum all the pixes in each channel
    std = torch.pow(std / count, 0.5)
    return mean, std


class DetectionImageTransformOnly:
    '''Constructs a wrapper object to apply the given image transformation only to the image, not the bounding boxes'''
    def __init__(self, img_transform):
        self.img_transform = img_transform
    def __call__(self, input):
        img = self.img_transform(input[0])
        target = input[1]
        return img, target

class DetectionRandomFlip:
    '''Random Flip for image and bounding boxes.'''

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, input):
        if random.random() < self.prob:
            image, target = input
            # Flip target
            image = TF.hflip(image)
            if isinstance(image,Image.Image):
                img_width = image.width
            else:
                img_width = image.shape[2]
            # Adjust bounding boxes
            for i, (xmin,ymin,xmax,ymax) in enumerate(target["boxes"]):
                old_xmax = int(xmax)
                target["boxes"][i][2] = img_width - xmin     # Adjust xmax
                target["boxes"][i][0] = img_width - old_xmax # Adjust xmin
            return image, target
        else:
            return input

class DetectionRandomResizedCrop:
    '''Random Resize and Crop image and adjust bounding boxes accordingly.'''

    def __init__(self, resize_size=(352,640), scale=(1.0,1.0), ratio=(1.0,1.0)):
        self.resize_size = resize_size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, input):
        image, target = input

        # 1. Get parameters for RandomResizedCrop (top, left, height, width)
        i, j, h, w = v2.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)

        # 2. Apply the same crop to both
        image = TF.crop(image, i, j, h, w)

        # 3. Resize both to final size
        image = TF.resize(image, self.resize_size, interpolation=TF.InterpolationMode.BILINEAR)
        scale_y = self.resize_size[0] / h
        scale_x = self.resize_size[1] / w

        # Adjust bounding boxes and skipping the ones that land outside the crop area
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for idx, (xmin,ymin,xmax,ymax) in enumerate(target["boxes"]):
            # Ensure new boxes do not overflow new image (max/min) and scale it.
            xmin = max(0, min(xmin-j, w)) * scale_x
            ymin = max(0, min(ymin-i, h)) * scale_y
            xmax = max(0, min(xmax-j, w)) * scale_x
            ymax = max(0, min(ymax-i, h)) * scale_y

            # Skip bounding boxes that landed outside the crop area
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(target["labels"][idx])
            areas.append(target["area"][idx])
            iscrowd.append(target["iscrowd"][idx])

        new_target = {
            'boxes':    torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'image_id': target["image_id"],
            'area':     torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd':  torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
        }

        return (image, new_target)
