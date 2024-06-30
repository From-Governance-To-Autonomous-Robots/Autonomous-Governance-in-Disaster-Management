import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np 

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
