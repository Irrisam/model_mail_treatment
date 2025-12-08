from transformers import CamembertTokenizer, CamembertForSequenceClassification
from sklearn.metrics import roc_curve
import torch
import json
import sys
from preprocess_multi import preprocess


DATA_FILE = "pi_5_multi_classes.jsonl"
MODEL_DIR = "is_ip_multi"
device = "cpu"

model = CamembertForSequenceClassification.from_pretrained(
    MODEL_DIR).to(device)
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model.eval()


def softmax(x):
    return torch.softmax(x, dim=1)


labels = []
probs = []

with open(DATA_FILE, "r") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            text = preprocess(obj["subject"] + " " + obj.get("body", ""))
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
            p = softmax(logits)[0]
            probs.append(p.max().item())
            labels.append(obj["label"])

print("Min prob:", min(probs), "| Max prob:", max(probs))
print("Average confidence:", sum(probs)/len(probs))
