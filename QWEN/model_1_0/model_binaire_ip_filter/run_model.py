import json
from model_pipeline import classify_email
from test_is_ip import is_ip
from attachment_extractor import extract_attachments_text
from preprocess import preprocess


def predict_ip(subject: str, body: str, attachments_text: str = "") -> float:
    text = f"{subject}\n{body}\n{attachments_text}"
    text = preprocess(text)
    return float(is_ip(text))


def predict_category(subject: str, body: str, attachments_text: str = ""):
    return classify_email(subject, body, attachments_text)


def analyze_email(email_id: int, subject: str, body: str):

    attachments_text = extract_attachments_text(email_id)

    prob_ip = predict_ip(subject, body, attachments_text)
    print(f"Probabilité IP: {prob_ip:.2f}")

    if prob_ip < 0.60:
        print("Email NON IP ")
        return {
            "email_id": email_id,
            "is_ip": False,
            "probability": prob_ip,
            "labels": ["not_ip"]
        }

    categories = predict_category(subject, body, attachments_text)

    for c in categories:
        print(f" - {c['category']} (score={c['confidence']})")

    return {
        "email_id": email_id,
        "is_ip": True,
        "probability": prob_ip,
        "labels": categories
    }


# if __name__ == "__main__":

#     email_id = 123
#     subject = "Demande de changement d'adresse"
#     body = "Bonjour, je souhaite mettre à jour mon adresse postale."

#     result = analyze_email(email_id, subject, body)

#     print("\n=== Résultat JSON ===")
#     print(json.dumps(result, indent=2, ensure_ascii=False))
