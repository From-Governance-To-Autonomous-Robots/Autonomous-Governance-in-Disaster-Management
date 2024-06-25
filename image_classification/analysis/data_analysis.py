import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os

def perform_data_analysis(data_path, save_dir, label_columns):
    # Load the dataset
    data = pd.read_csv(data_path, sep='\t')

    # Summary statistics
    summary = data.describe(include='all')
    summary_str = summary.to_string()
    print(summary_str)
    wandb.log({'summary_statistics': wandb.Html(summary_str.replace('\n', '<br>'))})

    # Distribution of categorical columns
    for column in label_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=column)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{column}_distribution.png'))
        wandb.log({f'{column}_distribution': wandb.Image(os.path.join(save_dir, f'{column}_distribution.png'))})

    # Log the dataframe to WandB
    wandb.log({'dataset': wandb.Table(dataframe=data)})
