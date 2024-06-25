import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
import os 

class MultimodalDataset(Dataset):
    def __init__(self, csv_file,dataset_directory, text_columns, image_column, text_target_column, image_target_column, tokenizer_name='bert-base-uncased', max_seq_length=128, transform=None):
        self.dataset_dir = dataset_directory
        self.data = pd.read_csv(os.path.join(dataset_directory,csv_file), sep='\t')
        self.text_columns = text_columns
        self.image_column = image_column
        self.text_target_column = text_target_column
        self.image_target_column = image_target_column
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.transform = transform

        self.text_label_encoder = {text_target_column: LabelEncoder().fit(self.data[text_target_column])}
        self.image_label_encoder = {image_target_column: LabelEncoder().fit(self.data[image_target_column])}
        
        self.data[text_target_column] = self.text_label_encoder[text_target_column].transform(self.data[text_target_column])
        self.data[image_target_column] = self.image_label_encoder[image_target_column].transform(self.data[image_target_column])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        texts = self.data.iloc[idx][self.text_columns].values
        combined_text = " ".join(texts)

        encoded_text = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        text_input_ids = encoded_text['input_ids'].squeeze()
        text_attention_mask = encoded_text['attention_mask'].squeeze()

        img_path = self.data.iloc[idx][self.image_column]
        image = Image.open(os.path.join(self.dataset_dir,img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text_label = self.data.iloc[idx][self.text_target_column]
        image_label = self.data.iloc[idx][self.image_target_column]

        return text_input_ids, text_attention_mask, image, text_label, image_label

# Example transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
