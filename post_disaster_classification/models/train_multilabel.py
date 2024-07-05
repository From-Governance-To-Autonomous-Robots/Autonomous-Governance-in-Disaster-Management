import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pdb

def plot_confusion_matrix(cm, class_names, title_prefix="Confusion Matrix"):
    figs = []
    for i, matrix in enumerate(cm):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['False', 'True'], yticklabels=['False', 'True'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f"{title_prefix} - {class_names[i]}")
        figs.append(fig)
    return figs

def log_predictions_table(phase, images, ground_truths, predictions, class_names, threshold=0.5):
    table = wandb.Table(columns=["Image", "Ground Truth", "Prediction"])
    for img, gt, pred in zip(images, ground_truths, predictions):
        gt_labels = [class_names[i] for i, val in enumerate(gt) if val == 1]
        pred_labels = [class_names[i] for i, val in enumerate(pred) if val >= threshold]
        table.add_data(wandb.Image(img), ",".join(gt_labels), ",".join(pred_labels))
    wandb.log({f"{phase}/sample_predictions": table})

def save_metrics_report(report, save_path):
    with open(save_path, 'w') as file:
        json.dump(report, file, indent=4)

def denormalize(image_tensor, mean, std):
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    denormalized_image = image_tensor * std + mean
    return denormalized_image

def log_classification_report(val_report, label_map):
    for class_name, value in val_report.items():
        wandb.log({f"Validation/{class_name}_f1_score": value['f1-score']})
        wandb.log({f"Validation/{class_name}_precision": value['precision']})
        wandb.log({f"Validation/{class_name}_recall": value['recall']})

    wandb.log({"Validation/macro_f1_score": val_report['macro avg']['f1-score']})
    wandb.log({"Validation/macro_recall": val_report['macro avg']['recall']})
    wandb.log({"Validation/macro_precision": val_report['macro avg']['precision']})
    wandb.log({"Validation/weighted_f1_score": val_report['weighted avg']['f1-score']})
    wandb.log({"Validation/weighted_recall": val_report['weighted avg']['recall']})
    wandb.log({"Validation/weighted_precision": val_report['weighted avg']['precision']})

def train_model(model, train_loader, val_loader, criterion, optimizer, config, save_dir, class_names, mean, std):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_f1 = 0.0
    best_val_loss = np.inf
    best_model_epoch = 0
    best_model_path = ""
    patience = 5
    remaining_before_early_stopping = 0

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
        
        model.eval()
        val_loss = 0
        val_targets = []
        val_preds = []
        val_images = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(outputs.cpu().numpy())
                val_images.extend(images.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_preds = (np.array(val_preds) > 0.5).astype(int)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_report = classification_report(val_targets, val_preds, target_names=class_names, output_dict=True)
        print('Val Report\n')
        print(val_report)
        val_cm = multilabel_confusion_matrix(val_targets, val_preds)
        print('Val CM\n')
        print(val_cm)

        wandb.log({"Validation/loss": avg_val_loss})
        log_classification_report(val_report, {i: cls for i, cls in enumerate(class_names)})

        cm_figs = plot_confusion_matrix(val_cm, class_names=class_names)
        for i, cm_fig in enumerate(cm_figs):
            wandb.log({f"Validation/confusion_matrix_{class_names[i]}": wandb.Image(cm_fig)})
            plt.close(cm_fig)
        
        sample_images = val_images[:10]
        sample_images = [np.transpose(denormalize(img, mean, std), (1, 2, 0)) for img in sample_images]
        sample_images = [(img * 255).astype(np.uint8) for img in sample_images]
        log_predictions_table("Validation", sample_images, val_targets[:10], val_preds[:10], class_names)

        if remaining_before_early_stopping <= patience:
            remaining_before_early_stopping +=1
            if (val_f1 >= best_val_f1) and (avg_val_loss < best_val_loss) and (avg_val_loss > avg_train_loss):
                best_model_epoch = epoch
                best_val_f1 = val_f1
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                
                best_report_path = os.path.join(save_dir, 'best_metrics_report.json')
                save_metrics_report(val_report, best_report_path)

        if remaining_before_early_stopping > patience:
            if best_val_loss < np.inf:
                print(f'Early Stopping @ {epoch}')
                print(f'Best Model Saved @ {best_model_epoch}')
                print(f'Best Val Loss @ {best_val_loss}')
                break
            else:
                remaining_before_early_stopping = 0
