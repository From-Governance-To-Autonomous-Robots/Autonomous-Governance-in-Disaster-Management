import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import numpy as np
import os
import json
from PIL import Image
import torchvision.transforms as transforms

def denormalize(image_tensor, mean, std):
    mean = np.array(mean).reshape(-1, 1, 1)
    std = np.array(std).reshape(-1, 1, 1)
    denormalized_image = image_tensor * std + mean
    return denormalized_image

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def log_predictions_table(phase, texts, images, ground_truths, predictions, class_names):
    table = wandb.Table(columns=["Text", "Image", "Ground Truth", "Prediction"])
    for text, img, gt, pred in zip(texts, images, ground_truths, predictions):
        table.add_data(text, wandb.Image(img), class_names[gt], class_names[pred])
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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, config, save_dir, class_names,mean,std,original_val_texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_f1 = 0.0
    best_val_loss = np.inf
    best_model_epoch = 0
    best_model_path = ""
    patience = 5
    remaining_before_early_stopping = 0
    
    # Save the label map
    label_map_path = os.path.join(save_dir, 'label_map.json')
    label_map = save_label_map(class_names, label_map_path)
    wandb.save(label_map_path)

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        for texts, images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, images)
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
        val_images = []
        
        with torch.no_grad():
            for texts, images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                texts, images, labels = texts.to(device), images.to(device), labels.to(device)
                outputs = model(texts, images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_targets.extend(labels.cpu().numpy())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_texts.extend(texts.cpu().numpy())
                val_images.extend(images.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        val_report = classification_report(val_targets, val_preds, output_dict=True)
        val_cm = confusion_matrix(val_targets, val_preds)

        wandb.log({"Validation/loss": avg_val_loss})
        log_classification_report(val_report, class_names)
        
        cm_fig = plot_confusion_matrix(val_cm, class_names=class_names)
        wandb.log({"Validation/confusion_matrix": wandb.Image(cm_fig)})
        plt.close(cm_fig)
        
        # Log a sample of batch predictions
        sample_texts = original_val_texts[:10]
        # sample_texts = val_texts[:10]
        sample_images = val_images[:10]
        sample_images = [np.transpose(denormalize(img, mean, std), (1, 2, 0)) for img in sample_images]
        sample_images = [(img * 255).astype(np.uint8) for img in sample_images]
        log_predictions_table("Validation", sample_texts, sample_images, val_targets[:10], val_preds[:10], class_names)

        if remaining_before_early_stopping <= patience:
            remaining_before_early_stopping += 1
            if (val_report['macro avg']['f1-score'] > best_val_f1) and (avg_val_loss < best_val_loss) and (avg_val_loss >= avg_train_loss):
                best_model_epoch = epoch
                best_val_f1 = val_report['macro avg']['f1-score']
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                
                best_report_path = os.path.join(save_dir, 'best_metrics_report.json')
                save_metrics_report(val_report, best_report_path)
                
        
        if remaining_before_early_stopping > patience:
            print(f'Early Stopping @ {epoch}')
            print(f'Best Model Saved @ {best_model_epoch}')
            print(f'Best Val Loss @ {best_val_loss}')
            break


def run_inference(model, train_loader, val_loader, criterion, optimizer, scheduler, config, save_dir, class_names,mean,std,original_val_texts):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    train_dataset_infer = []
    val_dataset_infer = []
    with torch.no_grad():
        for texts, images, labels, img_path,tweet_text in tqdm(train_loader, desc=f"Running Inference on training data"):
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            outputs = model(texts, images)
            _, predictions = torch.max(outputs, 1)
            prediction_conf = torch.softmax(outputs, dim=1).cpu().numpy()
            for i in range(len(labels)):
                row = {
                    "tweet_text": tweet_text[i],
                    "img_path": img_path[i],
                    "prediction": predictions[i].cpu().numpy(),
                    "ground_truth": labels[i].cpu().numpy(),
                    "prediction_conf": prediction_conf[i]
                }
                train_dataset_infer.append(row)
        
        for texts, images, labels, img_path,tweet_text in tqdm(val_loader, desc=f"Running Inference on validation data"):
            texts, images, labels = texts.to(device), images.to(device), labels.to(device)
            outputs = model(texts, images)
            _, predictions = torch.max(outputs, 1)
            prediction_conf = torch.softmax(outputs, dim=1).cpu().numpy()
            for i in range(len(labels)):
                row = {
                    "tweet_text": tweet_text[i],
                    "img_path": img_path[i],
                    "prediction": predictions[i].cpu().numpy(),
                    "ground_truth": labels[i].cpu().numpy(),
                    "prediction_conf": prediction_conf[i]
                }
                val_dataset_infer.append(row)
                
    # Save inference results to CSV
    train_df = pd.DataFrame(train_dataset_infer)
    val_df = pd.DataFrame(val_dataset_infer)
    train_df.to_csv(os.path.join(save_dir, "train_inference_results.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val_inference_results.csv"), index=False)