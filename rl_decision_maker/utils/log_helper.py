import pandas as pd 
import os 
import wandb
import numpy as np 

def log_to_csv(cumulative_data,csv_file_path):
    columns=["env_id","tree_score", "isTreeCorrectlyAnswered", "currentEpisode", "currentStep", "tree_id", "currentStepReward", "number_of_correctly_answered_trees", "number_of_wrongly_answered_trees", "number_of_times_additional_data_requested"]
    if not os.path.exists(csv_file_path):
        mode = 'w' 
    else:
        mode = 'a'  
    new_data_df = pd.DataFrame([cumulative_data], columns=columns)
    
    if mode == 'w':
        new_data_df.to_csv(csv_file_path, mode=mode, index=False)
    else:
        new_data_df.to_csv(csv_file_path, mode=mode, index=False, header=False)
        

def log_results_table_to_wandb(csv_file_path,prefix='Train'):
    columns=["env_id","tree_score", "isTreeCorrectlyAnswered", "currentEpisode", "currentStep", "tree_id", "currentStepReward", "number_of_correctly_answered_trees", "number_of_wrongly_answered_trees", "number_of_times_additional_data_requested"]
    df = pd.read_csv(csv_file_path)
    results_table = wandb.Table(columns=columns)
    for index, row in df.iterrows():
        results_table.add_data(*row)
    wandb.log({f"{prefix}/results_table": results_table})
    
def calculate_aggregate_stats(infos,key):
    _mean = np.array([ info[i][key] for i, info in enumerate(infos)]).mean()
    _std = np.array([ info[i][key] for i, info in enumerate(infos)]).std()
    return _mean,_std

def log_aggregate_stats(infos,key,log_string,step):
    _mean ,_std = calculate_aggregate_stats(infos,key)
    wandb.log({f"mean_{log_string}":_mean,"step":step + 1})
    wandb.log({f"std_c{log_string}":_std,"step":step + 1})
    
def calculate_aggregate_stats(arr):
    _mean = np.array(arr).mean()
    _std = np.array(arr).std()
    return _mean,_std

def log_aggregate_stats(infos,key,log_string,step):
    _mean ,_std = calculate_aggregate_stats(infos[key])
    wandb.log({f"mean_{log_string}":_mean,"step":step + 1})
    wandb.log({f"std_c{log_string}":_std,"step":step + 1})