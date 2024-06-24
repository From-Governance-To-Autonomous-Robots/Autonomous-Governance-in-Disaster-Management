import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def preprocess_text(text, tokenizer_name, max_seq_length):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded_text['input_ids'].squeeze(), encoded_text['attention_mask'].squeeze()
