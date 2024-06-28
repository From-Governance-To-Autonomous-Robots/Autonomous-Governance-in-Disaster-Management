import pandas as pd 
import os 

def load_and_combine_tsv_data(tsv_files:list,dataset_dir:str):
    file_paths = [os.path.join(dataset_dir, file) for file in tsv_files]
    dfs = [pd.read_csv(file_path, sep='\t') for file_path in file_paths]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    cleaned_df = combined_df[combined_df['image_damage'] != 'dont_know_or_cant_judge']
    
    cleaned_df.to_csv(os.path.join(dataset_dir,'whole_dataset.csv'), index=False)
    
    return cleaned_df,os.path.join(dataset_dir,'whole_dataset.csv')