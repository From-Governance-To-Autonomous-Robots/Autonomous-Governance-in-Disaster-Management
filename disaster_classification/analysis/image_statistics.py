import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def log_image_statistics(train_data, val_data, target_column):
    """
    Calculates image statistics, creates bar plots for text length distributions, and logs them to wandb.

    Args:
        train_data (pd.DataFrame): The training dataset.
        val_data (pd.DataFrame): The validation dataset.
        text_column (str): The column name containing text data.
        target_column (str): The column name containing target labels.
    """

    def plot_class_distribution(data, dataset_type):
        plt.figure(figsize=(12, 8))
        plot = sns.countplot(x=target_column, data=data)
        plt.title(f'Class Distribution - {dataset_type}')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.grid(True)
        
        # Add count annotations
        for p in plot.patches:
            plot.annotate(format(p.get_height(), '.0f'), 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha = 'center', va = 'center', 
                          xytext = (0, 9), 
                          textcoords = 'offset points')
        
        plt.tight_layout()
        wandb.log({f"{dataset_type}/class_distribution": wandb.Image(plt)})
        plt.close()

    plot_class_distribution(train_data, 'Train')
    plot_class_distribution(val_data, 'Validation')

