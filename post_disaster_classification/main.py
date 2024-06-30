import os
import argparse
import wandb
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils.utils import load_config, plot_class_distribution,save_class_labels_mapping
from data.dataset import CustomImageDataset
from models.resnet import CustomResNet
from models.train import train_model
from models.inference import load_model,get_class_labels_mapping,evaluate_and_log

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
    train_dataset_dir = os.path.join(config['paths']['dataset_dir_path'],config['paths']['phases'][0])
    val_dataset_dir = os.path.join(config['paths']['dataset_dir_path'], config['paths']['phases'][1])
    train_dataset = CustomImageDataset(train_dataset_dir, transform=train_transform,ignore_classes=config['model_training_parameters']['classes_to_ignore'],permitted_background_thresh=config['model_training_parameters']['permited_background_thresh'])
    val_dataset = CustomImageDataset(val_dataset_dir, transform=val_transform,ignore_classes=config['model_training_parameters']['classes_to_ignore'],permitted_background_thresh=config['model_training_parameters']['permited_background_thresh'])
    
    # Plot class distribution
    plot_class_distribution(train_dataset, "Train Class Distribution", "Train/class_distribution")
    plot_class_distribution(val_dataset, "Validation Class Distribution", "Validation/class_distribution")

    # Save class labels mapping
    labels_mapping_path = os.path.join(model_dir, 'class_labels_mapping.json')
    save_class_labels_mapping(train_dataset.classes, labels_mapping_path)
    wandb.save(labels_mapping_path)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Define the model
    model = CustomResNet(len(train_dataset.classes))
    
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model_training_parameters']['learning_rate'])
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, config['model_training_parameters'], model_dir, train_dataset.classes,mean=mean, std=std)

    # Load the best model for evaluation
    device='cuda' if torch.cuda.is_available() else 'cpu'
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    model = load_model(best_model_path, len(train_dataset.classes),device)
    class_mapping = get_class_labels_mapping(labels_mapping_path)

    # Run evaluation and log results to wandb
    evaluate_and_log(config, model, class_mapping,device )
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train for Post Disaster - Satellite and Drone Images')
    parser.add_argument('-config', type=str, default='/home/aaimscadmin/git-repo/cloud_repo/post_disaster_classification/configs/drone_config.yaml')
    args = parser.parse_args()
    
    main(args)
