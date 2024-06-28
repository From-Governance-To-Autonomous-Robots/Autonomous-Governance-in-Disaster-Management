import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.files = []
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            for file in os.listdir(cls_dir):
                self.files.append((os.path.join(cls_dir, file), cls))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, cls = self.files[idx]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.classes.index(cls)
        return image, label
