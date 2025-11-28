# import torch
# from transformers import CamembertTokenizer, CamembertModel
# import torch.nn.functional as F
# from labels import LABELS
# from preprocess import preprocess

# DEVICE = "cpu"

# tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
# encoder = CamembertModel.from_pretrained("camembert-base").to(DEVICE)
# encoder.eval()

# @torch.no_grad()
# def embed(text):
#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         max_length=256
#     ).to(DEVICE)
#     return encoder(**inputs).last_hidden_state.mean(dim=1)

# THRESHOLD = 0.70

# def classify_email(subject: str, body: str, attachments_text: str = ""):
#     clean_text = preprocess(f"{subject}. {body} {attachments_text}")
#     email_vec = embed(clean_text)

#     scores = []
#     for label in LABELS:
#         label_vec = embed(label)
#         sim = float(F.cosine_similarity(email_vec, label_vec))
#         scores.append({"category": label, "confidence": round(sim, 2)})

#     scores.sort(key=lambda x: x["confidence"], reverse=True)
#     best = scores[0]

#     if best["confidence"] < THRESHOLD:
#         return [{"category": "Autre / Non PI", "confidence": best["confidence"]}]

#     return scores[:3]

import torch
import torch.nn.functional as F
from transformers import CamembertTokenizer, CamembertForSequenceClassification, CamembertModel
from labels import LABELS
from preprocess import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "is_ip_model"   # dossier créé par train_is_ip.py

# ----------------------------
# 1️⃣ CHARGER TOKENIZER + MODELE FINE-TUNE
# ----------------------------
tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)

# Charger le modèle ENTIER finetuné
clf_model = CamembertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
clf_model.eval()

# Extraire le BACKBONE Camembert (sans la couche de classification)
encoder = clf_model.roberta   # Camembert = wrapper autour de RoBERTa
encoder.eval()

# ----------------------------
# 2️⃣ FONCTION D’EMBEDDING
# ----------------------------
@torch.no_grad()
def embed(text: str):
    """Retourne l'embedding moyen (CLS pooling améliorable si besoin)."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    # utiliser l'encodeur finetuné
    outputs = encoder(**inputs)
    hidden = outputs.last_hidden_state  # [batch, seq, hidden]

    return hidden.mean(dim=1)  # pooling moyen → [batch, hidden]


# ----------------------------
# 3️⃣ CLASSIFICATION PAR SIMILARITÉ COSINUS
# ----------------------------

THRESHOLD = 0.70

def classify_email(subject: str, body: str, attachments_text: str = ""):
    clean_text = preprocess(f"{subject}. {body} {attachments_text}")
    email_vec = embed(clean_text)

    scores = []
    for label in LABELS:
        label_vec = embed(label)
        sim = float(F.cosine_similarity(email_vec, label_vec))
        scores.append({"category": label, "confidence": round(sim, 2)})

    scores.sort(key=lambda x: x["confidence"], reverse=True)
    best = scores[0]

    if best["confidence"] < THRESHOLD:
        return [{"category": "Autre / Non PI", "confidence": best["confidence"]}]

    return scores[:3]
