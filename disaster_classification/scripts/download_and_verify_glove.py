import os
import requests
import zipfile
import numpy as np
from tqdm import tqdm

def download_glove(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Download the file with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    zip_path = os.path.join(save_path, 'glove.twitter.27B.zip')

    with open(zip_path, 'wb') as f:
        for data in tqdm(response.iter_content(1024), total=total_size//1024, unit='KB'):
            f.write(data)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_path)
    
    print(f"Downloaded and extracted GloVe embeddings to {save_path}")

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def verify_glove_embeddings(embeddings_index):
    # Test with a sample word
    test_word = 'hello'
    if test_word in embeddings_index:
        print(f"Embedding for '{test_word}': {embeddings_index[test_word]}")
    else:
        print(f"'{test_word}' not found in embeddings.")

if __name__ == "__main__":
    url = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"
    save_path = "glove_embeddings"
    
    download_glove(url, save_path)
    
    # Load and verify GloVe embeddings
    glove_file = os.path.join(save_path, 'glove.twitter.27B.50d.txt')
    embeddings_index = load_glove_embeddings(glove_file)
    verify_glove_embeddings(embeddings_index)
