import torch
import matplotlib.pyplot as plt
from visualization.detect.plot_dataset import draw_boxes_on_fig_ax
import wandb
import numpy as np

# Colors for each category
OBJECT_CLASS_COLOR = {
    0: np.array([0.0, 0.0, 0.0]),    # background  -> black
    1: np.array([1.0, 0.0, 0.0]),    # barrier  -> red
    2: np.array([0.0, 1.0, 0.0]),    # person   -> gree
    3: np.array([0.0, 0.5, 1.0]),    # refuel   -> blue
    4: np.array([1.0, 1.0, 0.0]),    # vehicle  -> yellow
}

def predict_targets(model, samples):
    '''Function to produce predictions using a given model and a list of (image ; ground truth) tuples'''
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Stack the images and the ground_truth
    images_list, targets_list = zip(*samples)
    images = torch.stack(images_list)

    # Setup the model for prediction
    model.eval()
    model.to(_device)
    with torch.no_grad():

        # Move the input images to the computing device and perform the inference
        images_device = images.to(_device)
        predictions = model(images_device)

    return images_list, targets_list, predictions


def prediction(images, ground_truths, predictions, output_path, categories, enable = False, N=6):
    
    '''Plot the predict images'''
    
    # Set some pamameters and create the matplotlib figure
    size_ratio = 5
    ratio_adj = 640/352
    fig = plt.figure(figsize=(2*ratio_adj*size_ratio, size_ratio * N))
    ax_array = fig.subplots(N, 2)
    
    for i in range(len(images)):

        # Load the image and its respective mask
        img, ground_truth, prediction = images[i], ground_truths[i], predictions[i]
        img_id = int(ground_truth["image_id"])

        # First, let's permute the image dimensions because Matplotlib imshow expexts HxWxC
        img_permuted = img.permute(1,2,0).cpu().numpy()

        # Then, the image and the ground truth objects
        ax_array[i,0].imshow(img_permuted)
        ax_array[i,0].set_title(f"Ground Truth (Idx: {img_id})")
        ax_array[i,0].axis('off')
        draw_boxes_on_fig_ax(ax_array[i,0], ground_truth["boxes"], ground_truth["labels"], categories, colors=OBJECT_CLASS_COLOR)

        # Finally, the image and the predicted objects
        ax_array[i,1].imshow(img_permuted)
        ax_array[i,1].set_title(f"Predicted (score)")
        ax_array[i,1].axis('off')
        draw_boxes_on_fig_ax(ax_array[i,1], prediction["boxes"], prediction["labels"], categories, OBJECT_CLASS_COLOR, prediction["scores"], score_threshold=0.5)

    plt.tight_layout()
    
    #Save image
    if enable:
        wandb.log({"predictions": wandb.Image(fig)})
    else:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()
