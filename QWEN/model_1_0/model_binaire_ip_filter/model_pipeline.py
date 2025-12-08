import torch
import torch.nn.functional as F
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from preprocess import preprocess

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = "is_ip_model_2"
THRESHOLD_PI = 0.51
TEMPERATURE = 0.7

CLASS_THRESHOLDS = {
    "Brevets": 0.45,
    "Marques_DM": 0.50,
    "Contentieux": 0.60,
    "Noms de domaine et Surveillance": 0.45,
    "Extensions": 0.50
}

FINE_LABELS = {
    "Brevets": ["Brevets"],
    "Marques_DM": ["Marques", "Dessins & modèles"],
    "Contentieux": ["Contrefaçon", "Oppositions", "Litiges PI"],
    "Noms de domaine et Surveillance": ["Noms de domaine", "Surveillance"],
    "Extensions": ["Extension territoriale"]
}

tokenizer = CamembertTokenizer.from_pretrained(MODEL_DIR)
clf_model = CamembertForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(DEVICE)
clf_model.eval()

encoder = clf_model.roberta
encoder.eval()


@torch.no_grad()
def embed(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    ).to(DEVICE)

    outputs = encoder(**inputs)
    hidden = outputs.last_hidden_state
    return hidden.mean(dim=1)


@torch.no_grad()
def predict_is_pi(subject: str, body: str, attachments: str = ""):
    text = preprocess(f"{subject} {body} {attachments}")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    logits = clf_model(**inputs).logits
    prob = torch.softmax(logits / TEMPERATURE, dim=1)[0][1].item()

    decision = "PI" if prob >= THRESHOLD_PI else "Non-PI"
    return round(prob, 3), decision


def classify_email(subject: str, body: str, attachments: str = ""):
    prob, decision = predict_is_pi(subject, body, attachments)

    if decision == "Non-PI":
        return {
            "decision": decision,
            "confidence": prob,
            "top_categories": []
        }

    clean_text = preprocess(f"{subject}. {body} {attachments}")
    email_vec = embed(clean_text)

    scores = []
    for macro_label, variants in FINE_LABELS.items():
        sub_scores = []
        for v in variants:
            v_vec = embed(v)
            sim = float(F.cosine_similarity(email_vec, v_vec))
            sub_scores.append(sim)

        max_sim = max(sub_scores)
        scores.append({"category": macro_label,
                      "confidence": round(max_sim, 3)})

    scores.sort(key=lambda x: x["confidence"], reverse=True)
    best = scores[0]

    calibrated_decision = (
        f"{best['category']} (hautement certain)"
        if best["confidence"] >= CLASS_THRESHOLDS[best["category"]]
        else f"{best['category']} (faible certitude)"
    )

    return {
        "decision": decision,
        "confidence": prob,
        "top_categories": scores[:3],
        "class_decision": calibrated_decision
    }
