import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from preprocess import preprocess
from sklearn.model_selection import train_test_split

DATA_FILE = "pi_labels_dataset.jsonl"
MODEL_NAME = "camembert-base"
SAVE_DIR = "is_ip_model"
BATCH_SIZE = 4
EPOCHS = 5
LR = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
model = CamembertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)


class EmailDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = preprocess(item["subject"] + " " + item["body"])
        label = 1 if item["label"] == "PI" else 0
        enc = tokenizer(
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc.input_ids.squeeze(), enc.attention_mask.squeeze(), torch.tensor(label)


rows = []
with open(DATA_FILE, "r") as f:
    for line in f:
        rows.append(json.loads(line))

train_data, val_data = train_test_split(rows, test_size=0.2, shuffle=True)

train_ds = EmailDataset(train_data)
val_ds = EmailDataset(val_data)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

optim = AdamW(model.parameters(), lr=LR)


def accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, att, y in data_loader:
            x, att, y = x.to(device), att.to(device), y.to(device)
            logits = model(x, att).logits
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


for epoch in range(EPOCHS):
    model.train()
    losses = 0

    for x, att, y in train_dl:
        x, att, y = x.to(device), att.to(device), y.to(device)
        optim.zero_grad()
        logits = model(x, att).logits
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        optim.step()
        losses += loss.item()

    val_acc = accuracy(model, val_dl)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {losses:.3f} - Val Acc: {val_acc:.3f}")

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("Model saved in:", SAVE_DIR)
