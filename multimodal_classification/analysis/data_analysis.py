import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os

def perform_data_analysis(data_path, save_dir,label_columns):
    # Load the dataset
    data = pd.read_csv(data_path, sep='\t')

    # Summary statistics
    summary = data.describe(include='all')
    summary_str = summary.to_string()
    print(summary_str)
    wandb.log({'summary_statistics': wandb.Html(summary_str.replace('\n', '<br>'))})

    # Distribution of categorical columns and their confidence scores
    # label_columns = ['text_info', 'image_info', 'text_human', 'image_human', 'image_damage']
    for column in label_columns:
        # Plot the distribution of labels
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x=column)
        plt.title(f'{column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{column}_distribution.png'))
        wandb.log({f'{column}_distribution': wandb.Image(os.path.join(save_dir, f'{column}_distribution.png'))})

        # Plot the distribution of confidence scores for each label
        conf_column = column + '_conf'
        for label in data[column].unique():
            plt.figure(figsize=(10, 6))
            sns.histplot(data[data[column] == label][conf_column].dropna(), bins=30, kde=True)
            plt.title(f'{column} - {label} Confidence Score Distribution')
            plt.xlabel(f'{conf_column}')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{column}_{label}_conf_distribution.png'))
            wandb.log({f'{column}_{label}_conf_distribution': wandb.Image(os.path.join(save_dir, f'{column}_{label}_conf_distribution.png'))})

    # Distribution of text lengths
    data['text_length'] = data['tweet_text'].apply(lambda x: len(x.split()) if pd.notnull(x) else 0)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=30, kde=True)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_length_distribution.png'))
    wandb.log({'text_length_distribution': wandb.Image(os.path.join(save_dir, 'text_length_distribution.png'))})

    # Log the dataframe to WandB
    wandb.log({'dataset': wandb.Table(dataframe=data)})

# # Example usage
# perform_data_analysis('data/hurricane_irma_final_data.tsv', 'logs')
