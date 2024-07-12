import os
import argparse
import pandas as pd 
import wandb
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from utils.utils import load_config, plot_label_distribution,save_class_labels_mapping
from models.resnet import MultiLabelResNet
from models.train_multilabel import run_inference
from PIL import Image
import numpy as np

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None, combine_class=None,ignore_classes=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.combine_class = combine_class
        self.ignore_classes = ignore_classes
        self.classes = list(self.data.columns[1:])

    def _combine_classes(self):
        for key, value in self.combine_class.items():
            if key in self.data.columns and value in self.data.columns:
                self.data.loc[self.data[key] == 1, value] = 1
                self.data.drop(columns=[key], inplace=True)
        self.classes = list(self.data.columns[1:])  # Update classes after combining
    
    def _ignore_classes(self):
        if len(self.ignore_classes) > 0:
            for value in self.ignore_classes:
                self.data.drop(columns=[value], inplace=True)
        self.classes = list(self.data.columns[1:])  # Update classes after dropping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        labels = self.data.iloc[idx, 1:].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels,img_path

def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'], name=f"run_{config['paths']['task']}_inference")
    # Data transformations
    mean = [0.485, 0.456, 0.406]
    std =  [0.229, 0.224, 0.225]
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create model save directory
    model_dir = f"{config['paths']['task']}_saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    train_dataset = MultiLabelDataset(os.path.join(model_dir,f"train_{config['paths']['task']}.csv"), transform=val_transform)
    val_dataset = MultiLabelDataset(os.path.join(model_dir,f"val_{config['paths']['task']}.csv"), transform=val_transform)
    
    # # Plot class distribution
    plot_label_distribution(train_dataset, "Train Class Distribution", "Train/class_distribution")
    plot_label_distribution(val_dataset, "Validation Class Distribution", "Validation/class_distribution")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=False,num_workers=4,pin_memory=True)
    
    # Define the model
    base_model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = MultiLabelResNet(base_model, len(train_dataset.classes))
    
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_training_parameters']['learning_rate'])
    
    model.load_state_dict(torch.load(config['inference']['model_path']))
    
    run_inference(model, train_loader, val_loader, criterion, optimizer, config['model_training_parameters'], model_dir, train_dataset.classes,mean=mean, std=std)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train for Post Disaster - Satellite and Drone Images')
    parser.add_argument('-config', type=str, default='/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/post_disaster_classification/configs/satellite_config.yaml')
    args = parser.parse_args()
    
    main(args)
