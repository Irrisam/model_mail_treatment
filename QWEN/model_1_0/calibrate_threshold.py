import json
import torch
from sklearn.metrics import roc_curve, f1_score
from preprocess import preprocess
from transformers import CamembertTokenizer, CamembertForSequenceClassification

DATA_FILE = "pi_labels_dataset.jsonl"
MODEL_DIR = "is_ip_model"
device = "cpu"

tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

def get_prob(text: str):
    text = preprocess(text)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.softmax(logits, dim=1)[0][1].item()

probs = []
labels = []

with open(DATA_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        text = item["subject"] + " " + item["body"]
        prob = get_prob(text)
        probs.append(prob)
        labels.append(1 if item["label"] == "PI" else 0)

fpr, tpr, roc_thresh = roc_curve(labels, probs)
youden = tpr - fpr
best_idx = youden.argmax()
best_threshold = roc_thresh[best_idx]

possible_thresh = [i/100 for i in range(1,100)]
f1_scores = [f1_score(labels, [1 if p>=th else 0 for p in probs]) for th in possible_thresh]
best_f1_thresh = possible_thresh[f1_scores.index(max(f1_scores))]

print("=== Recommended Thresholds ===")
print(f"Youden's J best threshold : {best_threshold:.3f}")
print(f"F1-max threshold          : {best_f1_thresh:.3f}")

print("!!! Performance at Youden threshold !!!!!")
preds = [1 if p>=best_threshold else 0 for p in probs]
acc = sum([int(p==l) for p,l in zip(preds, labels)]) / len(labels)
print(f"Accuracy: {acc:.3f}")
