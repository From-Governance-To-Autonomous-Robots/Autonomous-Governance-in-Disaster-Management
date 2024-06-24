import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel

class MultimodalModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=2):
        super(MultimodalModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final layer
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.fc = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_labels)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.cnn(image)
        text_features = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output
