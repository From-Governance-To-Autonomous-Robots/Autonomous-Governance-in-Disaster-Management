import os
import argparse
import pandas as pd 
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from utils.utils import load_config, plot_label_distribution,save_class_labels_mapping
from data.dataset import MultiLabelDataset
from models.resnet import MultiLabelResNet
from models.train_multilabel import train_model

def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'], name=f"run_{config['paths']['task']}")
    # Data transformations
    mean = [0.485, 0.456, 0.406]
    std =  [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Create model save directory
    model_dir = f"{config['paths']['task']}_saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load datasets
    train_csv_path = os.path.join(config['original_data']['root_dir'],'train','logs/dataset.csv')
    val_csv_path = os.path.join(config['original_data']['root_dir'],'val','logs/dataset.csv')
    
    if 'combine_classes' in config['model_training_parameters'] and config['model_training_parameters']['combine_classes']:
        combine_classes = config['model_training_parameters']['combine_classes']
    else:
        combine_classes = None
        
    if 'classes_to_ignore' in config['model_training_parameters'] and config['model_training_parameters']['classes_to_ignore']:
        classes_to_ignore = config['model_training_parameters']['classes_to_ignore']
    else:
        classes_to_ignore = None
    print('Combine CLasses : ',combine_classes)
    train_dataset = MultiLabelDataset(train_csv_path, transform=train_transform,combine_class=combine_classes,ignore_classes=classes_to_ignore,save_dir=os.path.join(model_dir,f"train_{config['paths']['task']}.csv"))
    val_dataset = MultiLabelDataset(val_csv_path, transform=val_transform,combine_class=combine_classes,ignore_classes=classes_to_ignore,save_dir=os.path.join(model_dir,f"val_{config['paths']['task']}.csv"))
    
    # # Plot class distribution
    plot_label_distribution(train_dataset, "Train Class Distribution", "Train/class_distribution")
    plot_label_distribution(val_dataset, "Validation Class Distribution", "Validation/class_distribution")

    # Save class labels mapping
    labels_mapping_path = os.path.join(model_dir, 'class_labels_mapping.json')
    save_class_labels_mapping(train_dataset.classes, labels_mapping_path)
    wandb.save(labels_mapping_path)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=True,num_workers=16,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=False,num_workers=16,pin_memory=True)
    
    # Define the model
    base_model = models.resnet50(pretrained=True)

    # Replace the final fully connected layer
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = MultiLabelResNet(base_model, len(train_dataset.classes))
    
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_training_parameters']['learning_rate'])
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, config['model_training_parameters'], model_dir, train_dataset.classes,mean=mean, std=std)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train for Post Disaster - Satellite and Drone Images')
    parser.add_argument('-config', type=str, default='/home/julian/git-repo/juliangdz/GovernanceIRP/Autonomous-Governance-in-Disaster-Management/post_disaster_classification/configs/satellite_config.yaml')
    args = parser.parse_args()
    
    main(args)
