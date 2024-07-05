import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np 
import pandas as pd
import torch
import pdb

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None,ignore_classes=None,permitted_background_thresh=0.8):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([cls for cls in os.listdir(data_dir) if cls not in ignore_classes ])
        self.files = self.clean_up_for_only_valid_images(data_dir,ignore_classes,permitted_background_thresh)
                    
    def clean_up_for_only_valid_images(self,data_dir:str,ignore_classes:list,permitted_background_thresh:float):
        """
        Will clean the classes that are in ignore and also use permitted background threshold to filter classes with more black rather than context
        """
        files = []
        self.classes = sorted([cls for cls in os.listdir(data_dir) if cls not in ignore_classes ])
        self.files = []
        for cls in self.classes:
            if cls not in ignore_classes:
                cls_dir = os.path.join(data_dir, cls)
                for file in os.listdir(cls_dir):
                    file_path = os.path.join(cls_dir, file)
                    ab_img = np.array(Image.open(file_path).convert("RGB"))
                    pixel_val , counts = np.unique(ab_img,return_counts=True)
                    percentage_of_background_in_image = counts[list(pixel_val).index(0)] / np.sum(counts)
                    if percentage_of_background_in_image < permitted_background_thresh:
                        files.append((file_path, cls))
        return files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, cls = self.files[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.classes.index(cls)
        return image, label


class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, transform=None, combine_class=None,ignore_classes=None,save_dir='somthing'):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.combine_class = combine_class
        self.ignore_classes = ignore_classes
        self.classes = list(self.data.columns[1:])
        
        if ignore_classes:
            self._ignore_classes()
        
        if combine_class:
            self._combine_classes()
        
        self.data.to_csv(save_dir,index=False)

        # # Ensure all label columns are numeric and convert them to integers
        # for cls in self.classes:
        #     self.data[cls] = pd.to_numeric(self.data[cls], errors='coerce').fillna(0).astype(int)

    def _combine_classes(self):
        for key, value in self.combine_class.items():
            if key in self.data.columns and value in self.data.columns:
                self.data.loc[self.data[key] == 1, value] = 1
                self.data.drop(columns=[key], inplace=True)
        self.classes = list(self.data.columns[1:])  # Update classes after combining
    
    def _ignore_classes(self):
        if len(self.ignore_classes) > 0:
            for value in self.ignore_classes:
                self.data.drop(columns=[value], inplace=True)
        self.classes = list(self.data.columns[1:])  # Update classes after dropping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        labels = self.data.iloc[idx, 1:].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels