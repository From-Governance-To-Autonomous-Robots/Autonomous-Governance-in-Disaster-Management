import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from PIL import Image
import os
import json
import numpy as np
from models.resnet import CustomResNet
import torchvision.models as models

def load_model(model_path, num_classes, device):
    model = CustomResNet(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_class_labels_mapping(mapping_path):
    with open(mapping_path, 'r') as file:
        class_mapping = json.load(file)
    return {int(k): v for k, v in class_mapping.items()}

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def run_inference(model, image_path, transform, class_mapping, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Get the top k predictions, where k is the minimum of 5 and the number of classes
    k = min(5, len(class_mapping))
    top_prob, top_catid = torch.topk(probabilities, k)
    top_labels = [class_mapping[catid.item()] for catid in top_catid]
    return image, top_labels, top_prob.cpu().detach().numpy()

def get_mask_labels(mask, labels):
    unique_labels = np.unique(mask)
    label_names = [labels[idx] for idx in unique_labels]
    return label_names

def plot_predictions(image, top5_labels, top5_prob, mask_image, mask_labels, title):
    fig, ax = plt.subplots(1, 2, figsize=(14, 10))
    
    ax[0].imshow(image)
    ax[0].set_title("Image with Top-5 Predictions")
    ax[0].axis('off')

    # Position the text below the image
    for i, (label, prob) in enumerate(zip(top5_labels, top5_prob)):
        ax[0].text(0, 1.1 - (i + 1) * 0.1, f"{label}: {prob:.2f}", transform=ax[0].transAxes, fontsize=12, color='red', backgroundcolor='white')

    ax[1].imshow(mask_image, cmap='tab10')
    ax[1].set_title("Ground Truth Mask")
    ax[1].axis('off')
    ax[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label=label, 
                markerfacecolor='black', markersize=10) for label in mask_labels], loc='best')

    fig.suptitle(title)
    plt.tight_layout()
    return fig

def evaluate_and_log(config, model, class_mapping, device):
    transform = get_transforms()
    labels = config['paths']['evaluation_gt_labels']
    
    image_dir = config['paths']['evaluation_image_dir_path']
    mask_dir = config['paths']['evaluation_ground_truth_dir_path']

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name.split('.')[0] + f"_{config['paths']['label_file_prefix']}.png")

        image, top5_labels, top5_prob = run_inference(model, image_path, transform, class_mapping, device)
        
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask_labels = get_mask_labels(mask, labels)
        
        fig = plot_predictions(image, top5_labels, top5_prob, mask, mask_labels, f"Evaluation: {image_name}")
        wandb.log({f"Evaluation/{image_name}": wandb.Image(fig)})
        plt.close(fig)
