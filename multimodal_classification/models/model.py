import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_text_labels=2, num_image_labels=2):
        super(MultimodalModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final layer

        self.text_fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_text_labels)
        )

        self.image_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_image_labels)
        )

    def forward(self, text_input_ids, text_attention_mask, image):
        text_outputs = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_pooled_output = text_outputs.pooler_output
        text_logits = self.text_fc(text_pooled_output)

        image_features = self.cnn(image)
        image_logits = self.image_fc(image_features)

        return text_logits, image_logits
