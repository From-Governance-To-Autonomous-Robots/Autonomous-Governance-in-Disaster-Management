import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import numpy as np
import os
import json

class TextDataset(Dataset):
    def __init__(self, texts, labels, original_texts):
        self.texts = texts
        self.labels = labels
        self.original_texts = original_texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.original_texts[idx]
        )

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def log_predictions_table(phase, texts, ground_truths, predictions, class_names):
    table = wandb.Table(columns=["Text", "Ground Truth", "Prediction"])
    for text, gt, pred in zip(texts, ground_truths, predictions):
        table.add_data(text, class_names[gt], class_names[pred])
    wandb.log({f"{phase}/sample_predictions": table})

def save_metrics_report(report, save_path):
    with open(save_path, 'w') as file:
        json.dump(report, file, indent=4)

def save_label_map(class_names, save_path):
    label_map = {idx: class_name for idx, class_name in enumerate(class_names)}
    with open(save_path, 'w') as file:
        json.dump(label_map, file, indent=4)
    return label_map

def log_classification_report(val_report, label_map):
    for key, value in val_report.items():
        if key.isdigit():
            class_name = label_map[int(key)]
            wandb.log({f"Validation/{class_name}_f1_score": value['f1-score']})
            wandb.log({f"Validation/{class_name}_precision": value['precision']})
            wandb.log({f"Validation/{class_name}_recall": value['recall']})

    wandb.log({"Validation/accuracy": val_report['accuracy']})
    wandb.log({"Validation/macro_f1_score": val_report['macro avg']['f1-score']})
    wandb.log({"Validation/macro_recall": val_report['macro avg']['recall']})
    wandb.log({"Validation/macro_precision": val_report['macro avg']['precision']})
    wandb.log({"Validation/weighted_f1_score": val_report['weighted avg']['f1-score']})
    wandb.log({"Validation/weighted_recall": val_report['weighted avg']['recall']})
    wandb.log({"Validation/weighted_precision": val_report['weighted avg']['precision']})

def train_model(model, train_data, val_data, train_labels, val_labels, original_train_texts, original_val_texts, config, save_dir, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_dataset = TextDataset(train_data, train_labels, original_train_texts)
    val_dataset = TextDataset(val_data, val_labels, original_val_texts)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    best_val_f1 = 0.0
    best_model_path = ""
    best_report_path = ""

    # Save the label map
    label_map_path = os.path.join(save_dir, 'label_map.json')
    label_map = save_label_map(class_names, label_map_path)
    wandb.save(label_map_path)

    # early_stopping_patience = 10
    # early_stopping_counter = 0

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for texts, labels, orig_texts in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"Train/loss": avg_train_loss})
        
        # Validation
        model.eval()
        val_loss = 0
        val_targets = []
        val_preds = []
        val_texts = []
        
        with torch.no_grad():
            for texts, labels, orig_texts in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_texts.extend(orig_texts)
        
        avg_val_loss = val_loss / len(val_loader)
        # scheduler.step()
        val_report = classification_report(val_targets, val_preds, output_dict=True)
        val_cm = confusion_matrix(val_targets, val_preds)

        wandb.log({"Validation/loss": avg_val_loss})
        log_classification_report(val_report, label_map) 
        
        cm_fig = plot_confusion_matrix(val_cm, class_names=class_names)
        wandb.log({"Validation/confusion_matrix": wandb.Image(cm_fig)})
        plt.close(cm_fig)
        
        # Log a sample of batch predictions
        log_predictions_table("Validation", val_texts[:10], val_targets[:10], val_preds[:10], class_names)

        if val_report['macro avg']['f1-score'] > best_val_f1:
            best_val_f1 = val_report['macro avg']['f1-score']
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            
            best_report_path = os.path.join(save_dir, 'best_metrics_report.json')
            save_metrics_report(val_report, best_report_path)
            # early_stopping_counter = 0  # Reset the early stopping counter
        # else:
            # early_stopping_counter += 1

        
        # if early_stopping_counter >= early_stopping_patience:
        #     print("Early stopping triggered")
        #     break
