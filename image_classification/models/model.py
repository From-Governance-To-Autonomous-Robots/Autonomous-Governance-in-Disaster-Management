import torch
import torch.nn as nn
from torchvision import models

class ImageClassificationModel(nn.Module):
    def __init__(self, num_labels=2):
        super(ImageClassificationModel, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )

    def forward(self, image):
        return self.cnn(image)
