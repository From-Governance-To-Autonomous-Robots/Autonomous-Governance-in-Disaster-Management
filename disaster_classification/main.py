from data.data_cleaning import TextCleaning
from data.data_preprocess import split_and_save_dataset,apply_pad_sequences,apply_tokenizer,apply_label_encoding
from data.load_glove import load_glove_embeddings
from data.load_raw_data import load_and_combine_tsv_data
from analysis.text_statistics import log_text_statistics
from utils.utils import load_config
import argparse
import pandas as pd 
import os 
import nltk
import wandb
from models.bilstm import BiLSTM
from models.train import train_model

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def main(args):
    config = load_config(args.config)
    wandb.init(project=config['model_training_parameters']['wandb_project'],name=f"run_text_{config['paths']['task']}_train")
    
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
    train_data, val_data = split_and_save_dataset(filtered_dataset,text_target_column,config['model_training_parameters']['test_size'],config['model_training_parameters']['random_state'],log_dir,f"text_{config['paths']['task']}")
    
    # Calculate the max words, max_sequence length , max_features, the 95 the percentile for each and log it as bar plot with the count information, and log the plot to wandb . 
    # Do a distrbution plot for each train , and test on the task columns and how many rows of text data is available for each class in the task column , the bar plot will show the count information as well , and log to wandb the plot
    log_text_statistics(train_data,val_data,text_column="cleaned_text",target_column=text_target_column)
    
    # Apply Keras Tokenizer
    tokenizer, train_sequences = apply_tokenizer(train_data['cleaned_text'], config['model_training_parameters']['vocab_size'])
    _, val_sequences = apply_tokenizer(val_data['cleaned_text'], config['model_training_parameters']['vocab_size'])

    # Apply pad_sequences
    train_padded = apply_pad_sequences(train_sequences, config['model_training_parameters']['max_seq_len'])
    val_padded = apply_pad_sequences(val_sequences, config['model_training_parameters']['max_seq_len'])
    
    # Apply label encoding
    label_encoder, train_encoded_labels = apply_label_encoding(train_data[text_target_column])
    _, val_encoded_labels = apply_label_encoding(val_data[text_target_column])
    
    # Load the glove embeddings
    glove_file = config['paths']['glove_file']
    embedding_dim = config['model_training_parameters']['embedding_dim']
    embedding_matrix = load_glove_embeddings(glove_file, tokenizer.word_index, embedding_dim)

    #Build the BiLSTM model with PyTorch
    vocab_size = config['model_training_parameters']['vocab_size']
    hidden_dim = config['model_training_parameters']['hidden_dim']
    num_classes = len(set(train_encoded_labels))

    model = BiLSTM(vocab_size, embedding_dim, embedding_matrix, hidden_dim, num_classes)
    
    # train  loop - use tqdm progress bar - print the train loss, val loss and accuracy for train and val at each epoch , also before it enters into eval loop , it should run the clasification report for training and log the the train_accuracy, train_f1_score, train_recall, train_precision to wandb. Then it enters into validation and does clasification report - logs the val_loss, val_accuracy, val_f1_score, val_recall, val_precision. 
    # Also compute the confusion matrix and roc_auc , and create plots for this which is logged as wandb.Image to wandb. 
    # the model files need to be saved , so handle this as well . can create a new directory based on the task name and then save the model files there.
    # Training loop
    model_dir =f"text_{config['paths']['task']}_saved_models"
    os.makedirs(model_dir, exist_ok=True)
    class_names = label_encoder.classes_.tolist()

    train_model(model, train_padded, val_padded, train_encoded_labels, val_encoded_labels, train_data['cleaned_text'].tolist(), val_data['cleaned_text'].tolist(), config['model_training_parameters'], model_dir, class_names)

    wandb.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="unit test for text cleaning")
    parser.add_argument('-config', type=str,default='/home/aaimscadmin/git-repo/cloud_repo/disaster_classification/configs/human_config.yaml')
    args = parser.parse_args()
    
    main(args)