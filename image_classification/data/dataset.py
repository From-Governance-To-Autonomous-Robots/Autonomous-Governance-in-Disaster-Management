import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import torch
import os

class ImageClassificationDataset(Dataset):
    def __init__(self, csv_file, dataset_directory, image_column, target_column, transform=None):
        self.dataset_dir = dataset_directory
        self.data = pd.read_csv(os.path.join(dataset_directory, csv_file), sep='\t')
        self.image_column = image_column
        self.target_column = target_column
        self.transform = transform

        self.label_encoder = LabelEncoder().fit(self.data[self.target_column])
        self.data[self.target_column] = self.label_encoder.transform(self.data[self.target_column])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx][self.image_column]
        image = Image.open(os.path.join(self.dataset_dir, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx][self.target_column]
        return image, label

# Example transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
