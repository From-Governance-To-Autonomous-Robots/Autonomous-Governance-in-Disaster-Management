import os

class Config:
    # Paths
    TSV_FILES = [
        'annotations/california_wildfires_final_data.tsv',
        'annotations/hurricane_harvey_final_data.tsv',
        'annotations/hurricane_irma_final_data.tsv',
        'annotations/hurricane_maria_final_data.tsv',
        'annotations/iraq_iran_earthquake_final_data.tsv',
        'annotations/mexico_earthquake_final_data.tsv',
        'annotations/srilanka_floods_final_data.tsv'
    ]
    COMBINED_DATA_PATH = 'annotations/combined_data_human.tsv'
    TRAIN_DATA_PATH = 'annotations/train_data_human.tsv'
    VAL_DATA_PATH = 'annotations/val_data_human.tsv'
    DATASET_DIR_PATH = '/home/aaimscadmin/IRP_DATA/CrisisMMD_v2.0'
    MODEL_SAVE_DIR = 'saved_models_human'
    LOG_DIR = 'logs_human'
    
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 5

    # BERT
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_SEQ_LENGTH = 128

    # WandB
    WANDB_PROJECT = 'multimodal-classification'
    WANDB_NAME = 'training_human'

    # Target Columns for Training
    TEXT_COLUMNS = ['tweet_text','image_url']  # Specify which columns to use as input features
    IMAGE_COLUMN = 'image_path'
    TEXT_TARGET_COLUMN = 'text_human'
    IMAGE_TARGET_COLUMN = 'image_human'
    DROPPED_COLUMNS = ['tweet_id', 'image_id', 'text_info','text_info_conf', 'image_info','image_info_conf', 'image_damage','image_damage_conf']

    # Ensure directories exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR_PATH, exist_ok=True)