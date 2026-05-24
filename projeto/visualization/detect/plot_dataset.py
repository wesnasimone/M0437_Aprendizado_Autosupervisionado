import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import json

'''
Examples:

DATA_FOLDER         = os.path.join(os.getcwd(), 'data')
DET_DATASET_ZIPFILE = os.path.join(DATA_FOLDER, 'detection_640x360.zip')
DET_DATASET_FOLDER  = os.path.join(DATA_FOLDER, 'detection_640x360')

det_dataset_config = json.load(open(os.path.join(DET_DATASET_FOLDER, 'dataset_info.json')))

DET_CATEGORIES = {int(k)+1: v for k, v in det_dataset_config['class_map'].items()}
DET_CATEGORIES[0] = "background"

(1)
split = "train"
plot_random_seg_samples(images_folder = os.path.join(DET_DATASET_FOLDER, f'images/{split}'),
                        JSON_filename = os.path.join(DET_DATASET_FOLDER, f'{split}.json'),
                        categories = DET_CATEGORIES,
                        colors = OBJECT_CLASS_COLOR,
                        n_samples = 3)

(2)
from data_processing.detect.detect_transforms import DetectionImageTransformOnly, DetectionRandomResizedCrop

images_dir=os.path.join(DET_DATASET_FOLDER, f"images/train")
coco_filename = os.path.join(DET_DATASET_FOLDER, f"train.json")

with open(coco_filename) as f:
    coco_data = json.load(f)

transforms = T.Compose([DetectionRandomResizedCrop(resize_size=(352,640), scale=(0.5, 1.0), ratio=(3.0/4.0,4.0/3.0)),
                        DetectionImageTransformOnly(T.ToTensor())
                        ])

train_ds = DetectionDataset(images_dir=images_dir, coco_data=coco_data, transform=transforms)

# List of samples to inspect.
samples_to_plot = [0,95,4,6]  # Plot different samples

# Plot the samples
plot_det_dataset_samples(train_ds, samples_to_plot, categories=DET_CATEGORIES, colors=OBJECT_CLASS_COLOR)

'''

# Colors for each category
OBJECT_CLASS_COLOR = {
    0: np.array([0.0, 0.0, 0.0]),    # background  -> black
    1: np.array([1.0, 0.0, 0.0]),    # barrier  -> red
    2: np.array([0.0, 1.0, 0.0]),    # person   -> gree
    3: np.array([0.0, 0.5, 1.0]),    # refuel   -> blue
    4: np.array([1.0, 1.0, 0.0]),    # vehicle  -> yellow
}

def plot_random_seg_samples(images_folder, JSON_filename, categories, colors=OBJECT_CLASS_COLOR, n_samples=3):
    """ Plot n random samples from the dataset images folder (images_folder). """

    # Read the samples labels (boxes and labels) from the JSON file
    with open(JSON_filename) as f:
        coco_data = json.load(f)

    anns_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        anns_by_image.setdefault(img_id, []).append(ann)

    # Selecting N random samples
    sampled_images = random.sample(coco_data['images'], min(n_samples, len(coco_data['images'])))

    # Create the figure
    size_ratio = 10
    ratio_adj = 640/352
    fig = plt.figure(figsize=(1*ratio_adj*size_ratio, size_ratio * n_samples))
    ax_array = fig.subplots(n_samples, 1, squeeze=False)

    # Plot the n random samples as subfigures.
    for i, img_info in enumerate(sampled_images):

        # Read the image from file
        img_path = os.path.join(images_folder, img_info['file_name'])
        img = np.array(Image.open(img_path).convert('RGB'))

        # Plot the image
        ax_array[i,0].imshow(img)
        ax_array[i,0].axis('off')
        ax_array[i,0].set_title(f"Image: {img_info['file_name']} (id={img_info['id']})")

        # Plot the boxes
        for ann in anns_by_image.get(img_info['id'], []):
            x, y, w, h = ann['bbox']
            cat_id = int(ann['category_id'])+1 # Adjust to account for background ID
            color = colors.get(cat_id, (1, 1, 1))
            label = categories.get(cat_id, f'cls_{cat_id}')

            # Draw the box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax_array[i,0].add_patch(rect)

            # Place the label on top of the box
            ax_array[i,0].text(
                x, y - 4, label,
                fontsize=10, fontweight='bold',
                color='black',
                bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none')
            )

    # Show the figure
    fig.show()


def draw_boxes_on_fig_ax(ax, boxes, boxes_cat_id, categories, colors=OBJECT_CLASS_COLOR, scores=None, score_threshold=0.5):
    '''Draw boxes and respective labels on top on a matplotlib image.'''
    for j in range(len(boxes)):
        # Ignore boxes with low prediction scores (Used when showing samples predicted by the model)
        if (scores is not None) and (scores[j] < score_threshold):
            continue

        xmin, ymin, xmax, ymax = boxes[j].cpu().numpy()
        width, height = xmax - xmin, ymax - ymin
        cat_id = int(boxes_cat_id[j])
        color = colors.get(cat_id, (1, 1, 1))
        label = categories.get(cat_id, f'cls_{cat_id}')

        if scores is not None:
            label = label + f" ({scores[j]:.2f})"

        # Draw the box
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Place the label on top of the box
        ax.text(
            xmin, ymin - 4, label,
            fontsize=10, fontweight='bold',
            color='black',
            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor='none')
        )

def plot_det_dataset_samples(dataset, sample_list, categories, colors):
    '''Plot the samples from the dataset. Annotate the bounding boxes using the categories and colors dictionaries'''
    # Create the figure
    size_ratio = 10
    ratio_adj = 640/352
    n = len(sample_list)
    fig = plt.figure(figsize=(1*ratio_adj*size_ratio, size_ratio * n))
    ax_array = fig.subplots(n, 1, squeeze=False)

    for i, idx in enumerate(sample_list):
        img, target = dataset[idx]

        # Plot the image
        ax_array[i,0].imshow(img.permute(1,2,0))
        ax_array[i,0].axis('off')
        ax_array[i,0].set_title(f"IDX: {idx}")

        # Plot the boxes
        draw_boxes_on_fig_ax(ax_array[i,0], target["boxes"], target["labels"])

    # Show the figure
    fig.show()
