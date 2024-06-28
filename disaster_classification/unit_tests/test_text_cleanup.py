import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data.data_cleaning import TextCleaning
from data.load_raw_data import load_and_combine_tsv_data
from utils.utils import load_config
import argparse
import pandas as pd 

parser = argparse.ArgumentParser(description="unit test for text cleaning")
parser.add_argument('-config', type=str)
args = parser.parse_args()

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

config = load_config(args.config)
data, csv_path = load_and_combine_tsv_data(config['paths']['tsv_files'], dataset_dir=config['paths']['dataset_dir_path'])

# Initialize the TextCleaning class
text_cleaner = TextCleaning()

# Apply text cleaning
data['cleaned_text'] = data['tweet_text'].apply(text_cleaner.do_text_cleanup)

# Save the cleaned data
data.to_csv('test_cleaned_dataset.csv', index=False)
