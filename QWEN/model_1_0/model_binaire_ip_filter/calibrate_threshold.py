import json
import torch
from preprocess import preprocess
from sklearn.metrics import roc_curve, f1_score, accuracy_score
from transformers import CamembertTokenizer, CamembertForSequenceClassification

DATA_FILE = "pi_labels_dataset.jsonl"
MODEL_DIR = "is_ip_model"
device = "cpu"

tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
model = CamembertForSequenceClassification.from_pretrained(
    MODEL_DIR).to(device)
model.eval()


def get_prob(text):
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


probs, labels = [], []

with open(DATA_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        text = item["subject"] + " " + item["body"]
        labels.append(1 if item["label"] == "PI" else 0)
        probs.append(get_prob(text))

# ROC
fpr, tpr, thresholds = roc_curve(labels, probs)
youden = tpr - fpr
youden_idx = youden.argmax()
best_youden = thresholds[youden_idx]

# F1
candidates = [i/100 for i in range(1, 100)]
f1_scores = [f1_score(labels, [1 if p >= th else 0 for p in probs])
             for th in candidates]
best_f1 = candidates[f1_scores.index(max(f1_scores))]

# Accuracy
preds = [1 if p >= best_youden else 0 for p in probs]
acc = accuracy_score(labels, preds)

print("Youden optimal threshold :", round(best_youden, 3))
print("F1 optimal threshold     :", round(best_f1, 3))
print("Accuracy Youden        :", round(acc, 3))
print("Min prob =", round(min(probs), 3), "| Max prob =", round(max(probs), 3))
