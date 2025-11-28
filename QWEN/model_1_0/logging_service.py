import json
import hashlib
import os
from datetime import datetime

LOG_PATH = "logs/classification_log.jsonl"
os.makedirs("logs", exist_ok=True)

def hash_content(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def log_decision(email_id: int, subject: str, decision: str, confidence: float = None, labels=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "email_id": email_id,
        "decision": decision,
        "confidence_ip": round(confidence, 2) if confidence is not None else None,
        "labels": labels if labels else [],
        "subject_hash": hash_content(subject)
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
