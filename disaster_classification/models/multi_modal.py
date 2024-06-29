import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, embed_size, embedding_matrix, hidden_size, num_classes):
        super(MultimodalModel, self).__init__()
        # Text model (BiLSTM)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        
        # Image model (ResNet50)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256)
        
        # Combined layers
        self.fc1 = nn.Linear(hidden_size * 4 + 256, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, text, image):
        # Text forward
        h_embedding = self.embedding(text)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        text_features = torch.cat((avg_pool, max_pool), 1)
        
        # Image forward
        image_features = self.resnet(image)
        
        # Combine features
        combined_features = torch.cat((text_features, image_features), 1)
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
