import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def split_and_save_dataset(dataset, target_column, test_size, random_state, output_dir,task):
    if not os.path.exists(os.path.join(output_dir, f'train_{task}_dataset.csv')):
        train_data, val_data = train_test_split(
            dataset, test_size=test_size, random_state=random_state, stratify=dataset[target_column]
        )

        os.makedirs(output_dir, exist_ok=True)
        
        train_path = os.path.join(output_dir, f'train_{task}_dataset.csv')
        val_path = os.path.join(output_dir, f'val_{task}_dataset.csv')
        
        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
    else:
        train_path = os.path.join(output_dir, f'train_{task}_dataset.csv')
        val_path = os.path.join(output_dir, f'val_{task}_dataset.csv')
        
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        
    print(f"Training data saved to {train_path}")
    print(f"Validation data saved to {val_path}")
    
    return train_data, val_data


def apply_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    return tokenizer, sequences

def apply_pad_sequences(sequences, max_sequence_length):
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences


def apply_label_encoding(labels):
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return label_encoder, encoded_labels