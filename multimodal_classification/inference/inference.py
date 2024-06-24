import torch
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
from models.model import MultimodalModel
from config import Config

def load_model(model_path, bert_model_name, num_labels):
    model = MultimodalModel(bert_model_name=bert_model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def preprocess_text(text, tokenizer, max_seq_length):
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True
    )
    return encoded_text['input_ids'], encoded_text['attention_mask']

def predict(image_path, text, model, tokenizer, max_seq_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = preprocess_image(image_path).to(device)
    input_ids, attention_mask = preprocess_text(text, tokenizer, max_seq_length)
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(image, input_ids, attention_mask)
        predicted_labels = torch.sigmoid(outputs).cpu().numpy()
    
    return predicted_labels

if __name__ == "__main__":
    model = load_model(Config.MODEL_SAVE_DIR + '/best_model.pth', Config.BERT_MODEL_NAME, num_labels=len(Config.TARGET_COLUMNS))
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
    
    # Example usage
    image_path = 'path/to/image.jpg'
    text = "Sample text for prediction"
    
    predicted_labels = predict(image_path, text, model, tokenizer, Config.MAX_SEQ_LENGTH)
    print(f'Predicted labels: {predicted_labels}')
