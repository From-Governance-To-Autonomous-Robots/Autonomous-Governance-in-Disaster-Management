import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
from data.dataset import MultimodalDataset, transform
from models.model import MultimodalModel
from config import Config

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_text_labels = []
    all_image_labels = []
    all_text_preds = []
    all_image_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for text_input_ids, text_attention_mask, image, text_labels, image_labels in tqdm(data_loader, desc="Validating"):
            text_input_ids, text_attention_mask = text_input_ids.to(device), text_attention_mask.to(device)
            image = image.to(device)
            text_labels, image_labels = text_labels.to(device), image_labels.to(device)

            text_outputs, image_outputs = model(text_input_ids, text_attention_mask, image)
            text_loss = criterion(text_outputs, text_labels)
            image_loss = criterion(image_outputs, image_labels)
            loss = text_loss + image_loss
            total_loss += loss.item() * text_input_ids.size(0)

            text_preds = torch.argmax(text_outputs, dim=1).cpu().numpy()
            image_preds = torch.argmax(image_outputs, dim=1).cpu().numpy()
            all_text_labels.extend(text_labels.cpu().numpy())
            all_image_labels.extend(image_labels.cpu().numpy())
            all_text_preds.extend(text_preds)
            all_image_preds.extend(image_preds)

    avg_loss = total_loss / len(data_loader.dataset)
    text_accuracy = accuracy_score(all_text_labels, all_text_preds)
    image_accuracy = accuracy_score(all_image_labels, all_image_preds)
    text_precision, text_recall, text_f1, _ = precision_recall_fscore_support(all_text_labels, all_text_preds, average='weighted')
    image_precision, image_recall, image_f1, _ = precision_recall_fscore_support(all_image_labels, all_image_preds, average='weighted')

    return avg_loss, text_accuracy, text_precision, text_recall, text_f1, image_accuracy, image_precision, image_recall, image_f1

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    train_dataset = MultimodalDataset(
        csv_file=config.TRAIN_DATA_PATH,
        text_columns=config.TEXT_COLUMNS,
        image_column=config.IMAGE_COLUMN,
        text_target_column=config.TEXT_TARGET_COLUMN,
        image_target_column=config.IMAGE_TARGET_COLUMN,
        tokenizer_name=config.BERT_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = MultimodalDataset(
        csv_file=config.VAL_DATA_PATH,
        text_columns=config.TEXT_COLUMNS,
        image_column=config.IMAGE_COLUMN,
        text_target_column=config.TEXT_TARGET_COLUMN,
        image_target_column=config.IMAGE_TARGET_COLUMN,
        tokenizer_name=config.BERT_MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_text_labels = len(train_dataset.text_label_encoder[config.TEXT_TARGET_COLUMN].classes_)
    num_image_labels = len(train_dataset.image_label_encoder[config.IMAGE_TARGET_COLUMN].classes_)
    model = MultimodalModel(bert_model_name=config.BERT_MODEL_NAME, num_text_labels=num_text_labels, num_image_labels=num_image_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_f1 = 0.0

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for text_input_ids, text_attention_mask, image, text_labels, image_labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            text_input_ids, text_attention_mask = text_input_ids.to(device), text_attention_mask.to(device)
            image = image.to(device)
            text_labels, image_labels = text_labels.to(device), image_labels.to(device)
            optimizer.zero_grad()

            text_outputs, image_outputs = model(text_input_ids, text_attention_mask, image)
            text_loss = criterion(text_outputs, text_labels)
            image_loss = criterion(image_outputs, image_labels)
            loss = text_loss + image_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * text_input_ids.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, text_accuracy, text_precision, text_recall, text_f1, image_accuracy, image_precision, image_recall, image_f1 = evaluate_model(model, val_loader, criterion, device)

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth'))

        if text_f1 + image_f1 > best_val_f1:
            best_val_f1 = text_f1 + image_f1
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))

        # Log metrics
        wandb.log({
            'epoch': epoch+1,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'text_accuracy': text_accuracy,
            'text_precision': text_precision,
            'text_recall': text_recall,
            'text_f1': text_f1,
            'image_accuracy': image_accuracy,
            'image_precision': image_precision,
            'image_recall': image_recall,
            'image_f1': image_f1
        })

        logging.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Text Acc: {text_accuracy:.4f}, Text Prec: {text_precision:.4f}, Text Recall: {text_recall:.4f}, Text F1: {text_f1:.4f}, Image Acc: {image_accuracy:.4f}, Image Prec: {image_precision:.4f}, Image Recall: {image_recall:.4f}, Image F1: {image_f1:.4f}")
