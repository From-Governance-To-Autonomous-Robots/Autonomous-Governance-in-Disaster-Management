from data.data_cleaning import TextCleaning
from data.data_preprocess import split_and_save_dataset
from data.load_raw_data import load_and_combine_tsv_data
from analysis.text_statistics import log_text_statistics
from utils.utils import load_config
import argparse
import pandas as pd 
import os 
import nltk
import wandb

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'],name=f"run_{config['paths']['task']}_data")
    
    dataset, whole_dataset_csv_path = load_and_combine_tsv_data(config['paths']['tsv_files'], dataset_dir=config['paths']['dataset_dir_path'])
    
    # Create folders
    log_dir = os.path.join(config['paths']['dataset_dir_path'],'logs')
    os.makedirs(log_dir,exist_ok=True)
    
    # Initialize the TextCleaning class
    text_cleaner = TextCleaning()
    dataset['cleaned_text'] = dataset['tweet_text'].apply(text_cleaner.do_text_cleanup)
    dataset.to_csv(os.path.join(log_dir,'cleaned_dataset.csv'), index=False)
    
    # Filter to use only the dataset for the task columns 
    text_target_column = f"text_{config['paths']['task']}"
    filtered_dataset = dataset[['cleaned_text', text_target_column]]
    filtered_dataset = filtered_dataset.dropna(subset=[text_target_column])
    
    # task specific operations - drop rows with certain classes to reduce noise 
    if len(config['data']['classes_to_drop']) > 0:
        for class_drop in config['data']['classes_to_drop']:
            filtered_dataset = filtered_dataset[filtered_dataset[text_target_column] != class_drop]
    
    filtered_dataset.to_csv(os.path.join(log_dir,f"{config['paths']['task']}_dataset.csv"),index=False)
    
    # Split the dataset based on the classes, and save the val and train data as 2 csv , we will use these csv next time instead of doing a split again. 
    train_data, val_data = split_and_save_dataset(filtered_dataset,text_target_column,config['model_training_parameters']['test_size'],config['model_training_parameters']['random_state'],log_dir,config['paths']['task'])
    
    # Calculate the max words, max_sequence length , max_features, the 95 the percentile for each and log it as bar plot with the count information, and log the plot to wandb . 
    # Do a distrbution plot for each train , and test on the task columns and how many rows of text data is available for each class in the task column , the bar plot will show the count information as well , and log to wandb the plot
    log_text_statistics(train_data,val_data,text_column="cleaned_text",target_column=text_target_column)
    
    wandb.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="unit test for text cleaning")
    parser.add_argument('-config', type=str,default='/home/aaimscadmin/git-repo/cloud_repo/text_classification/configs/human_config.yaml')
    args = parser.parse_args()
    
    main(args)