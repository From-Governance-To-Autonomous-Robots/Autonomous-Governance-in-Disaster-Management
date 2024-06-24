import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
from data.dataset import MultimodalDataset,transform
from models.model import MultimodalModel
from config import Config
import numpy as np 

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, weights):
        loss = self.bce_loss(logits, targets)
        weighted_loss = loss * weights
        return weighted_loss.mean()

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for images, input_ids, attention_mask, labels, confidence_scores in tqdm(data_loader, desc="Validating"):
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
            labels, confidence_scores = labels.to(device), confidence_scores.to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels, confidence_scores)
            total_loss += loss.item() * images.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    avg_loss = total_loss / len(data_loader.dataset)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds) > 0.5

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, precision, recall, f1

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load data
    train_dataset = MultimodalDataset(
        csv_file=config.TRAIN_DATA_PATH,
        dataset_directory=config.DATASET_DIR_PATH,
        target_columns=config.TARGET_COLUMNS,
        transform=transform,
        tokenizer_name=config.BERT_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = MultimodalDataset(
        csv_file=config.VAL_DATA_PATH,
        dataset_directory=config.DATASET_DIR_PATH,
        target_columns=config.TARGET_COLUMNS,
        transform=transform,
        tokenizer_name=config.BERT_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_labels = len(config.TARGET_COLUMNS)
    model = MultimodalModel(bert_model_name=config.BERT_MODEL_NAME, num_labels=num_labels).to(device)
    criterion = WeightedBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_f1 = 0.0

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, input_ids, attention_mask, labels, confidence_scores in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
            labels, confidence_scores = labels.to(device), confidence_scores.to(device)
            optimizer.zero_grad()
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels, confidence_scores)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth'))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))

        # Log metrics
        wandb.log({
            'epoch': epoch+1,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })

        logging.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val Prec: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
