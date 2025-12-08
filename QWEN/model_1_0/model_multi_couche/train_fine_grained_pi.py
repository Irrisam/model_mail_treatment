import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score
from model_binaire_ip_filter.preprocess import preprocess

DATA_FILE = "pi_5_multi_classes.jsonl"
MODEL_NAME = "camembert-base"
SAVE_DIR = "is_ip_multi"
BATCH_SIZE = 4
EPOCHS = 4
LR = 2e-5
EARLY_STOP_PATIENCE = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
rows = []
with open(DATA_FILE, "r") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

LABELS = sorted(list({r["label"] for r in rows}))
label_to_id = {l: i for i, l in enumerate(LABELS)}
id_to_label = {i: l for l, i in label_to_id.items()}

print("Detected labels:", LABELS)


class EmailDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = preprocess(item["subject"] + " " + item.get("body", ""))
        label = label_to_id[item["label"]]
        enc = tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc.input_ids.squeeze(), enc.attention_mask.squeeze(), torch.tensor(label, dtype=torch.long)


train_data, val_data = train_test_split(
    rows, test_size=0.15, shuffle=True, stratify=[r["label"] for r in rows])

train_dl = DataLoader(EmailDataset(train_data),
                      batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(EmailDataset(val_data), batch_size=BATCH_SIZE)

# ---- Model ----
model = CamembertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS)
).to(device)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_dl) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

best_f1 = 0
no_improve_epochs = 0


def evaluate():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, att, y in val_dl:
            x, att, y = x.to(device), att.to(device), y.to(device)
            logits = model(x, att).logits
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred)
            trues.extend(y.cpu().numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds, average="weighted")


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, att, y in train_dl:
        x, att, y = x.to(device), att.to(device), y.to(device)
        optimizer.zero_grad()
        loss = model(x, att, labels=y).loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    acc, f1 = evaluate()
    print(
        f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss:.3f} | val_acc={acc:.3f} | val_f1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        no_improve_epochs = 0
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print("↳ Best model updated.")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print("↳ Early stopping triggered.")
            break

print("Training done. Best F1:", best_f1)
