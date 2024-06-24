import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
import os

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, dataset_directory, target_columns, transform=None, tokenizer_name='bert-base-uncased', max_seq_length=128):
        self.dataset_directory = dataset_directory
        self.data = pd.read_csv(os.path.join(dataset_directory,csv_file), sep='\t')
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length

        # Prepare the data
        self.target_columns = target_columns
        self.confidence_columns = [col + '_conf' for col in self.target_columns]
        
        # Encode labels
        self.label_encoders = {col: LabelEncoder().fit(self.data[col]) for col in self.target_columns}
        for col in self.target_columns:
            self.data[col] = self.label_encoders[col].transform(self.data[col])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(os.path.join(self.dataset_directory, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        text = self.data.iloc[idx]['tweet_text']
        labels = self.data.iloc[idx][self.target_columns].values.astype(np.float32)
        confidence_scores = self.data.iloc[idx][self.confidence_columns].values.astype(np.float32)
        
        encoded_text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        
        return image, encoded_text['input_ids'].squeeze(), encoded_text['attention_mask'].squeeze(), labels, confidence_scores

# Example transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
