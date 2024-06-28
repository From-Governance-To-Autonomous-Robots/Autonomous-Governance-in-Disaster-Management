import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    return fig

def log_predictions_table(phase, images, ground_truths, predictions):
    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])
    for img, gt, pred in zip(images, ground_truths, predictions):
        table.add_data(wandb.Image(img), gt, pred)
    wandb.log({f"{phase}/sample_predictions": table})

def train_model(model, train_image_paths, val_image_paths, train_labels, val_labels, transform, config, save_dir, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_dataset = ImageDataset(train_image_paths, train_labels, transform)
    val_dataset = ImageDataset(val_image_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    best_val_f1 = 0.0
    best_model_path = ""

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"Train/loss": avg_train_loss})
        
        # Validation
        model.eval()
        val_loss = 0
        val_outputs = []
        val_targets = []
        val_images = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_outputs.extend(outputs.cpu().detach().numpy())
                val_targets.extend(labels.cpu().detach().numpy())
                val_images.extend(images.cpu().detach().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_preds = np.argmax(val_outputs, axis=1)
        
        val_report = classification_report(val_targets, val_preds, output_dict=True)
        val_cm = confusion_matrix(val_targets, val_preds)
        # val_auc = roc_auc_score(val_targets, val_outputs, multi_class='ovo')

        wandb.log({"Validation/loss": avg_val_loss})
        wandb.log({"Validation/accuracy": val_report['accuracy']})
        wandb.log({"Validation/f1_score": val_report['weighted avg']['f1-score']})
        wandb.log({"Validation/recall": val_report['weighted avg']['recall']})
        wandb.log({"Validation/precision": val_report['weighted avg']['precision']})
        # wandb.log({"Validation/auc": val_auc})
        
        cm_fig = plot_confusion_matrix(val_cm, class_names=class_names)
        wandb.log({"Validation/confusion_matrix": wandb.Image(cm_fig)})
        plt.close(cm_fig)
        
        # Log a sample of batch predictions
        sample_images = val_images[:10]
        sample_images = [np.transpose(img, (1, 2, 0)) for img in sample_images]  # Convert from (C, H, W) to (H, W, C)
        sample_images = [(img * 255).astype(np.uint8) for img in sample_images]  # Convert to uint8 for wandb.Image
        log_predictions_table("Validation", sample_images, val_targets[:10], val_preds[:10])
        
        # Save the model after each epoch
        epoch_model_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_model_path)
        wandb.log({"epoch_model_path": epoch_model_path})

        if val_report['weighted avg']['f1-score'] > best_val_f1:
            best_val_f1 = val_report['weighted avg']['f1-score']
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        
        wandb.log({"best_model_path": best_model_path})
