import pandas as pd
from sklearn.model_selection import train_test_split
import os
from config import Config

def combine_tsv_files(tsv_files, combined_data_path, dropped_columns):
    combined_df = pd.DataFrame()
    for tsv_file in tsv_files:
        df = pd.read_csv(tsv_file, sep='\t')
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    combined_df = combined_df.drop(columns=dropped_columns)
    combined_df.to_csv(combined_data_path, sep='\t', index=False)
    return combined_df

def split_data(combined_df, train_data_path, val_data_path, target_columns):
    stratify_column = combined_df[target_columns].value_counts().min() >= 10
    if stratify_column:
        train_df, val_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df[target_columns], random_state=42)
    else:
        train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    train_df.to_csv(train_data_path, sep='\t', index=False)
    val_df.to_csv(val_data_path, sep='\t', index=False)

# if __name__ == "__main__":
#     combined_df = combine_tsv_files(Config.TSV_FILES, Config.COMBINED_DATA_PATH, Config.DROPPED_COLUMNS)
#     split_data(combined_df, Config.TRAIN_DATA_PATH, Config.VAL_DATA_PATH, Config.TEXT_TARGET_COLUMN)
#     split_data(combined_df, Config.TRAIN_DATA_PATH, Config.VAL_DATA_PATH, Config.IMAGE_TARGET_COLUMN)
