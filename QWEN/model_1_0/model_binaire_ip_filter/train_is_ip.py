import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from torch.optim import AdamW
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_linear_schedule_with_warmup

DATA_FILE = "pi_labels_dataset.jsonl"
MODEL_NAME = "camembert-base"
SAVE_DIR = "is_ip_model"
BATCH_SIZE = 4
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

tokenizer = CamembertTokenizer.from_pretrained(MODEL_NAME)
model = CamembertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    classifier_dropout=0.1
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

labels = [1 if r["label"] == "PI" else 0 for r in rows]

train_data, val_data = train_test_split(
    rows, test_size=0.2, shuffle=True, stratify=labels
)

train_ds = EmailDataset(train_data)
val_ds = EmailDataset(val_data)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)


def evaluate(model, data_loader):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, att, y in data_loader:
            x, att, y = x.to(device), att.to(device), y.to(device)
            logits = model(x, att).logits
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            gold.extend(y.cpu().tolist())

    return accuracy_score(gold, preds), f1_score(gold, preds)


best_f1 = 0
patience = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, att, y in train_dl:
        x, att, y = x.to(device), att.to(device), y.to(device)
        optim.zero_grad()

        logits = model(x, att).logits
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        scheduler.step()
        total_loss += loss.item()

    val_acc, val_f1 = evaluate(model, val_dl)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | loss={total_loss:.2f} | val_acc={val_acc:.3f} | val_f1={val_f1:.3f}"
    )

    # early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience = 0
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print("↳ Best model updated.")
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training completed. Best F1:", best_f1)
