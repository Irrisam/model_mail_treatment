import torch
from preprocess import preprocess
from transformers import CamembertTokenizer, CamembertForSequenceClassification

MODEL_DIR = "is_ip_model"
device = "cpu"

tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

def is_ip(text):
    clean = preprocess(text)
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=1)[0][1].item()
    return prob