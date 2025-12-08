from fastapi import FastAPI
from pydantic import BaseModel
from model_pipeline import classify_email
from test_is_ip import is_ip
from attachment_extractor import extract_attachments_text
from logging_service import log_decision
import json
import os
import re


REVIEW_LOWER = float(os.getenv("REVIEW_LOWER", "0.60"))
THRESHOLD_IP = float(os.getenv("THRESHOLD_IP", "0.73"))

app = FastAPI(
    title="IP Email Classifier",
    description="Classification offline emails PI",
    version="1.1.0",
)

CLIENT_CODE_PATTERN = re.compile(r"\b[A-Z]{2}\.[A-Z]{2,32}\.\d{3}\b")


def extract_client_code(text: str) -> str | None:
    if not text:
        return None

    text = text.upper()
    m = CLIENT_CODE_PATTERN.search(text)
    return m.group(0) if m else None


class Email(BaseModel):
    email_id: int
    subject: str
    body: str


@app.get("/logs")
async def get_logs():
    with open("logs/classification_log.jsonl", "r") as f:
        return [json.loads(l) for l in f.readlines()]


@app.post("/classify")
async def classify(email: Email):
    att_text = extract_attachments_text(email.email_id)
    text = f"{email.subject}\n{email.body}\n{att_text}"

    prob = is_ip(text)

    client_code = extract_client_code(text)

    if prob < REVIEW_LOWER:
        log_decision(
            email.email_id,
            email.subject,
            "not_ip",
            prob,
            labels=["not_ip"]
        )
        return {
            "labels": ["not_ip"],
            "filter": "not_ip",
            "confidence_ip": round(prob, 2),
            "client_code": client_code,
        }

    if REVIEW_LOWER <= prob < THRESHOLD_IP:
        log_decision(
            email.email_id,
            email.subject,
            "review",
            prob,
            labels=["review_needed"]
        )
        return {
            "labels": ["review_needed"],
            "filter": "review",
            "confidence_ip": round(prob, 2),
            "client_code": client_code,
        }

    labels = classify_email(email.subject, email.body)
    log_decision(
        email.email_id,
        email.subject,
        "ip",
        prob,
        labels=labels
    )
    return {
        "labels": labels,
        "filter": "ip",
        "confidence_ip": round(prob, 2),
        "client_code": client_code,
    }


MAILS_DIR = "mails_json"


@app.post("/batch")
async def batch():
    results = []
    processed_count = 0

    for file in os.listdir(MAILS_DIR):
        if not file.endswith(".json"):
            continue

        path = os.path.join(MAILS_DIR, file)
        processed_count += 1

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"⚠️ Fichier vide ignoré : {file}")
                    continue
                email = json.loads(content)
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read().strip()
                    if not content:
                        print(f"Fichier vide ignoré : {file}")
                        continue
                    email = json.loads(content)
                print(f"Encodage corrigé pour : {file}")
            except Exception:
                print(f"JSON invalide ignoré (encodage) : {file}")
                continue
        except json.JSONDecodeError:
            print(f"JSON cassé ou illisible : {file}")
            continue

        email_id = email.get("email_id")
        subject = email.get("subject") or email.get("email_object") or ""
        body = email.get("body") or email.get("email_body") or ""

        if not email_id:
            print(f"Ignoré (email_id manquant) : {file}")
            continue

        att_text = extract_attachments_text(email_id)
        prob = is_ip(f"{subject}\n{body}\n{att_text}")

        if prob < REVIEW_LOWER:
            result = {
                "email_id": email_id,
                "filter": "not_ip",
                "labels": ["not_ip"],
                "confidence_ip": round(prob, 2)
            }
            log_decision(email_id, subject, "not_ip", prob, labels=["not_ip"])

        elif prob < THRESHOLD_IP:
            result = {
                "email_id": email_id,
                "filter": "review",
                "labels": ["review_needed"],
                "confidence_ip": round(prob, 2)
            }
            log_decision(email_id, subject, "review",
                         prob, labels=["review_needed"])

        else:
            labels = classify_email(subject, body)
            result = {
                "email_id": email_id,
                "filter": "ip",
                "labels": labels,
                "confidence_ip": round(prob, 2)
            }
            log_decision(email_id, subject, "ip", prob, labels=labels)

        results.append(result)

    return {
        "processed_mails": processed_count,
        "classified_mails": len(results),
        "results": results
    }


@app.get("/health")
async def health():
    return {"status": "Service running"}
