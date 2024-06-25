import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import wandb
from data.dataset import ImageClassificationDataset, transform
from models.model import ImageClassificationModel

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, precision, recall, f1

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    train_dataset = ImageClassificationDataset(
        csv_file=config.TRAIN_DATA_PATH,
        dataset_directory=config.DATASET_DIR_PATH,
        image_column=config.IMAGE_COLUMN,
        target_column=config.IMAGE_TARGET_COLUMN,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    val_dataset = ImageClassificationDataset(
        csv_file=config.VAL_DATA_PATH,
        dataset_directory=config.DATASET_DIR_PATH,
        image_column=config.IMAGE_COLUMN,
        target_column=config.IMAGE_TARGET_COLUMN,
        transform=transform
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    num_labels = len(train_dataset.label_encoder.classes_)
    model = ImageClassificationModel(num_labels=num_labels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_f1 = 0.0

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, accuracy, precision, recall, f1 = evaluate_model(model, val_loader, criterion, device)

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, f'model_epoch_{epoch+1}.pth'))

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'best_model.pth'))

        # Log metrics
        wandb.log({
            'epoch': epoch+1,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        logging.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
