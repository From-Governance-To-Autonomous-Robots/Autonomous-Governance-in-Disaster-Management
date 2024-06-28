import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def log_text_statistics(train_data, val_data, text_column, target_column):
    """
    Calculates text statistics, creates bar plots for text length distributions, and logs them to wandb.

    Args:
        train_data (pd.DataFrame): The training dataset.
        val_data (pd.DataFrame): The validation dataset.
        text_column (str): The column name containing text data.
        target_column (str): The column name containing target labels.
    """

    def get_text_statistics(data, column):
        data['text_length'] = data[column].apply(lambda x: len(x.split()))
        max_words = data['text_length'].max()
        max_sequence_length = data['text_length'].mean() + 2 * data['text_length'].std()
        max_features = len(set(" ".join(data[column]).split()))
        percentile_95 = np.percentile(data['text_length'], 95)
        
        return max_words, max_sequence_length, max_features, percentile_95

    train_stats = get_text_statistics(train_data, text_column)
    val_stats = get_text_statistics(val_data, text_column)
    
    stats = {
        "Train/": {
            'max_words': train_stats[0],
            'max_sequence_length': train_stats[1],
            'max_features': train_stats[2],
            '95_percentile': train_stats[3]
        },
        "Validation/": {
            'max_words': val_stats[0],
            'max_sequence_length': val_stats[1],
            'max_features': val_stats[2],
            '95_percentile': val_stats[3]
        }
    }

    wandb.log(stats)

    def plot_text_length_distribution(data, dataset_type):
        plt.figure(figsize=(12, 8))
        sns.histplot(data['text_length'], bins=50, kde=True)
        plt.title(f'Text Length Distribution - {dataset_type}')
        plt.xlabel('Text Length')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        wandb.log({f"{dataset_type}/text_length_distribution": wandb.Image(plt)})
        plt.close()

    plot_text_length_distribution(train_data, 'Train')
    plot_text_length_distribution(val_data, 'Validation')

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

# Usage example:
# log_text_statistics(train_data, val_data, 'cleaned_text', text_target_column)
