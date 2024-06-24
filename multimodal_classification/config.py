import os

class Config:
    # Paths
    TRAIN_DATA_PATH = '/home/julian/datasets/crisis_mmd/train/data_image/hurricane_maria/hurricane_maria_final_data.tsv'
    VAL_DATA_PATH = '/home/julian/datasets/crisis_mmd/val/data_image/hurricane_irma/hurricane_irma_final_data.tsv'
    TRAIN_DATASET_DIR_PATH = '/home/julian/datasets/crisis_mmd/train'
    VAL_DATASET_DIR_PATH = '/home/julian/datasets/crisis_mmd/val'
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
    DROPPED_COLUMNS = ['tweet_id', 'image_id', 'image_path', 'text_human', 'image_human', 'image_damage']

    # Ensure directories exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
