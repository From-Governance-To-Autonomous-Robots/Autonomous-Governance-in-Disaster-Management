import wandb
import logging
from config import Config
from analysis.data_analysis import perform_data_analysis
from training.train import train_model
from data.prepare_data import combine_tsv_files, split_data

def setup_logging(log_dir):
    logging.basicConfig(filename=f'{log_dir}/training.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logging.info('Logging setup complete.')

def main():
    # Setup logging
    setup_logging(Config.LOG_DIR)
    
    # Initialize WandB
    wandb.init(project=Config.WANDB_PROJECT, name=Config.WANDB_NAME)
    
    # Combine TSV files and split the data
    combined_df = combine_tsv_files(Config.TSV_FILES, Config.COMBINED_DATA_PATH, Config.DROPPED_COLUMNS)
    split_data(combined_df, Config.TRAIN_DATA_PATH, Config.VAL_DATA_PATH, Config.TEXT_TARGET_COLUMN)
    split_data(combined_df, Config.TRAIN_DATA_PATH, Config.VAL_DATA_PATH, Config.IMAGE_TARGET_COLUMN)
    
    # Perform data analysis
    perform_data_analysis(Config.TRAIN_DATA_PATH, Config.LOG_DIR)
    perform_data_analysis(Config.VAL_DATA_PATH, Config.LOG_DIR)

    # Train the model
    train_model(Config)

    wandb.finish()

if __name__ == "__main__":
    main()
