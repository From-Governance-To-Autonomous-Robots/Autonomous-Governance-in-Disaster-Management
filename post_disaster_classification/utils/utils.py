import os
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import yaml
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def plot_class_distribution(data_dir, title, wandb_log_key):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    
    class_names = list(class_counts.keys())
    class_values = list(class_counts.values())

    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=class_names, y=class_values)
    
    # Display the count on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    wandb.log({wandb_log_key: wandb.Image(plt)})
    plt.close()


def save_class_labels_mapping(classes, save_path):
    class_mapping = {idx: class_name for idx, class_name in enumerate(classes)}
    with open(save_path, 'w') as file:
        json.dump(class_mapping, file, indent=4)
