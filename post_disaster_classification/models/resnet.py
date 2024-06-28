import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model
