import wandb
import logging
from config import Config
from analysis.data_analysis import perform_data_analysis
from training.train import train_model
import os 

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
    
    # Perform data analysis
    train_log_dir = os.path.join(Config.LOG_DIR,'train')
    os.makedirs(train_log_dir,exist_ok=True)
    perform_data_analysis(Config.TRAIN_DATA_PATH, train_log_dir)
    val_log_dir = os.path.join(Config.LOG_DIR,'val')
    os.makedirs(val_log_dir,exist_ok=True)
    perform_data_analysis(Config.VAL_DATA_PATH, val_log_dir)

    # # Train the model
    train_model(Config)

    wandb.finish()

if __name__ == "__main__":
    main()
