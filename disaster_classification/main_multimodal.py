import pdb
from data.data_cleaning import TextCleaning
from data.data_preprocess import split_and_save_dataset, apply_pad_sequences, apply_tokenizer, apply_label_encoding
from data.load_glove import load_glove_embeddings
from data.load_raw_data import load_and_combine_tsv_data
from analysis.text_statistics import log_text_statistics
from utils.utils import load_config
import argparse
import pandas as pd 
import os 
import nltk
import wandb
from models.multi_modal import MultimodalModel
from models.train_multimodal import train_model
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class CustomMultimodalDataset(Dataset):
    def __init__(self, text_data, image_paths, labels, transform):
        self.text_data = text_data
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.text_data[idx], dtype=torch.long)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, image, label


def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'], name=f"run_multimodal_{config['paths']['task']}_train")
    
    dataset, whole_dataset_csv_path = load_and_combine_tsv_data(config['paths']['tsv_files'], dataset_dir=config['paths']['dataset_dir_path'])
    
    # Create folders
    log_dir = os.path.join(config['paths']['dataset_dir_path'],'multimodal_logs',config['paths']['task'])
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize the TextCleaning class
    text_cleaner = TextCleaning()
    dataset['cleaned_text'] = dataset['tweet_text'].apply(text_cleaner.do_text_cleanup)
    dataset.to_csv(os.path.join(log_dir, 'cleaned_dataset.csv'), index=False)
    
    # Filter to use only the dataset for the task columns 
    text_target_column = f"image_{config['paths']['task']}"
    filtered_dataset = dataset[['cleaned_text', text_target_column, 'image_path']]
    filtered_dataset = filtered_dataset.dropna(subset=[text_target_column, 'image_path'])
    
    # Apply class relabeling if specified in the config
    if 'classes_to_relabel' in config['data'] and config['data']['classes_to_relabel']:
        classes_to_relabel = config['data']['classes_to_relabel']
        filtered_dataset[text_target_column] = filtered_dataset[text_target_column].replace(classes_to_relabel)
    
    # Task specific operations - drop rows with certain classes to reduce noise 
    if len(config['data']['classes_to_drop']) > 0:
        for class_drop in config['data']['classes_to_drop']:
            filtered_dataset = filtered_dataset[filtered_dataset[text_target_column] != class_drop]
    
    filtered_dataset.to_csv(os.path.join(log_dir, f"{config['paths']['task']}_dataset.csv"), index=False)
    
    # Split the dataset based on the classes, and save the val and train data as 2 csv , we will use these csv next time instead of doing a split again. 
    train_data, val_data = split_and_save_dataset(filtered_dataset, text_target_column, config['model_training_parameters']['test_size'], config['model_training_parameters']['random_state'], log_dir, text_target_column)
    
    # Calculate the max words, max_sequence length , max_features, the 95 the percentile for each and log it as bar plot with the count information, and log the plot to wandb . 
    log_text_statistics(train_data, val_data, text_column="cleaned_text", target_column=text_target_column)
    
    # Apply Keras Tokenizer
    tokenizer, train_sequences = apply_tokenizer(train_data['cleaned_text'], config['model_training_parameters']['vocab_size'])
    _, val_sequences = apply_tokenizer(val_data['cleaned_text'], config['model_training_parameters']['vocab_size'])

    # Apply pad_sequences
    train_padded = apply_pad_sequences(train_sequences, config['model_training_parameters']['max_seq_len'])
    val_padded = apply_pad_sequences(val_sequences, config['model_training_parameters']['max_seq_len'])
    
    # Apply label encoding
    label_encoder, train_encoded_labels = apply_label_encoding(train_data[text_target_column])
    _, val_encoded_labels = apply_label_encoding(val_data[text_target_column])
    
    # Load the glove embeddings
    glove_file = config['paths']['glove_file']
    embedding_dim = config['model_training_parameters']['embedding_dim']
    embedding_matrix = load_glove_embeddings(glove_file, tokenizer.word_index, embedding_dim)
    
    # Build the Multimodal model with PyTorch
    vocab_size = config['model_training_parameters']['vocab_size']
    hidden_dim = config['model_training_parameters']['hidden_dim']
    num_classes = len(set(train_encoded_labels))

    model = MultimodalModel(vocab_size, embedding_dim, embedding_matrix, hidden_dim, num_classes)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Define Image Transform
    train_transform = transforms.Compose([
        transforms.Resize(config['model_training_parameters']['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(config['model_training_parameters']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    # Create Dataloaders
    train_image_paths = [os.path.join(config['paths']['dataset_dir_path'],file_path) for file_path in list(train_data['image_path'])]
    val_image_paths = [os.path.join(config['paths']['dataset_dir_path'],file_path) for file_path in list(val_data['image_path'])]
    
    train_dataset = CustomMultimodalDataset(train_padded, train_image_paths, train_encoded_labels, train_transform)
    val_dataset = CustomMultimodalDataset(val_padded, val_image_paths, val_encoded_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model_training_parameters']['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Define Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['model_training_parameters']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    
    # Training loop
    model_dir = f"multimodal_{config['paths']['task']}_saved_models"
    os.makedirs(model_dir, exist_ok=True)
    class_names = label_encoder.classes_.tolist()
    
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config['model_training_parameters'], model_dir, class_names,mean,std,original_val_texts=val_data['cleaned_text'].tolist())

    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal Training Pipeline")
    parser.add_argument('-config', type=str, default='/home/aaimscadmin/git-repo/cloud_repo/disaster_classification/configs/human_config.yaml')
    args = parser.parse_args()
    
    main(args)
