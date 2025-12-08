import torch
import torch.nn.functional as F
from preprocess import preprocess
from model_pipeline import embed

PI_KEYWORDS = [
    "propriété intellectuelle", "brevet", "marque", "PCT", "EP",
    "INPI", "EUIPO", "WIPO", "opposition", "annuité",
    "dossier", "protection", "dépôt", "priorité",
    "extension territoriale", "classification", "notification"
]

THRESHOLD = 0.70

@torch.no_grad()
def is_ip_related(text: str):
    clean = preprocess(text)[:400]
    text_vec = embed(clean)

    sims = []
    for kw in PI_KEYWORDS:
        kw_vec = embed(kw)
        sims.append(float(F.cosine_similarity(text_vec, kw_vec)))

    max_sim = max(sims)

    return max_sim >= THRESHOLD, round(max_sim, 3)
