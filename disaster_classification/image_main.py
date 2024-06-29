from data.data_preprocess import split_and_save_dataset, apply_label_encoding
from data.load_raw_data import load_and_combine_tsv_data
from analysis.image_statistics import log_image_statistics
from utils.utils import load_config
import argparse
import pandas as pd 
import os 
import wandb
from models.resnet import ResNet50Model
from models.train_image import train_model

import torch
import torchvision.transforms as transforms

def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'], name=f"run_image_{config['paths']['task']}_train")
    
    dataset, whole_dataset_csv_path = load_and_combine_tsv_data(config['paths']['tsv_files'], dataset_dir=config['paths']['dataset_dir_path'])
    
    # Create folders
    log_dir = os.path.join(config['paths']['dataset_dir_path'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Filter to use only the dataset for the task columns 
    image_target_column = f"image_{config['paths']['task']}"
    filtered_dataset = dataset[['image_path', image_target_column]]
    filtered_dataset = filtered_dataset.dropna(subset=[image_target_column])
    
    # Task specific operations - drop rows with certain classes to reduce noise 
    if len(config['data']['classes_to_drop']) > 0:
        for class_drop in config['data']['classes_to_drop']:
            filtered_dataset = filtered_dataset[filtered_dataset[image_target_column] != class_drop]
    
    filtered_dataset.to_csv(os.path.join(log_dir, f"image_{config['paths']['task']}_dataset.csv"), index=False)
    
    # Split the dataset based on the classes, and save the val and train data as 2 csv files
    train_data, val_data = split_and_save_dataset(filtered_dataset, image_target_column, config['model_training_parameters']['test_size'], config['model_training_parameters']['random_state'], log_dir, f"image_{config['paths']['task']}")
    
    # Log image statistics
    log_image_statistics(train_data, val_data, target_column=image_target_column)
    
    # Apply label encoding
    label_encoder, train_encoded_labels = apply_label_encoding(train_data[image_target_column])
    _, val_encoded_labels = apply_label_encoding(val_data[image_target_column])
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Define image transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Build the ResNet50 model
    num_classes = len(label_encoder.classes_)
    model = ResNet50Model(num_classes=num_classes)
    
    # Training loop
    model_dir = f"image_{config['paths']['task']}_saved_models"
    os.makedirs(model_dir, exist_ok=True)
    class_names = label_encoder.classes_.tolist()
    
    train_image_paths = [os.path.join(config['paths']['dataset_dir_path'],file_path) for file_path in list(train_data['image_path'])]
    val_image_paths = [os.path.join(config['paths']['dataset_dir_path'],file_path) for file_path in list(val_data['image_path'])]

    train_model(model, train_image_paths, val_image_paths, train_encoded_labels, val_encoded_labels, train_transform,val_transform, config['model_training_parameters'], model_dir, class_names,mean,std)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Classifier Training")
    parser.add_argument('-config', type=str, default='/home/aaimscadmin/git-repo/cloud_repo/text_classification/configs/human_config.yaml')
    args = parser.parse_args()
    
    main(args)
