import os

class Config:
    # Paths
    TSV_FILES = [
        # 'annotations/california_wildfires_final_data.tsv',
        # 'annotations/hurricane_harvey_final_data.tsv',
        'annotations/hurricane_irma_final_data.tsv',
        'annotations/hurricane_maria_final_data.tsv'
        # 'annotations/iraq_iran_earthquake_final_data.tsv',
        # 'annotations/mexico_earthquake_final_data.tsv',
        # 'annotations/srilanka_floods_final_data.tsv'
    ]
    COMBINED_DATA_PATH = 'annotations/combined_data.tsv'
    TRAIN_DATA_PATH = 'annotations/train_data.tsv'
    VAL_DATA_PATH = 'annotations/val_data.tsv'
    DATASET_DIR_PATH = '/home/julian/datasets/crisis_mmd/train' 
    MODEL_SAVE_DIR = 'saved_models'
    LOG_DIR = 'logs'

    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 5

    # BERT
    BERT_MODEL_NAME = 'bert-base-uncased'
    MAX_SEQ_LENGTH = 128

    # WandB
    WANDB_PROJECT = 'multimodal-image-classification'
    WANDB_NAME = 'training'

    # Target Columns for Training
    TARGET_COLUMNS = ['text_info', 'image_info']
    DROPPED_COLUMNS = ['tweet_id', 'image_id', 'text_human','text_human_conf', 'image_human','image_human_conf', 'image_damage','image_damage_conf']

    # Ensure directories exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR_PATH, exist_ok=True)
