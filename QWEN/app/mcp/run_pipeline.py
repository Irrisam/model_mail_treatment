from app.mcp.model_client import classify_email_api
from app.mcp.lookup_sender import lookup_sender
from app.mcp.trust import compute_trust_score


async def run_pipeline(params):
    email_id = params.get("email_id")
    subject = params.get("subject", "")
    body = params.get("body", "")

    if not email_id:
        return {"error": "email_id is required"}

    # -----------------------
    # 1) Appel du modèle ML
    # -----------------------
    model_result = await classify_email_api(email_id, subject, body)

    # Exemple :
    # {
    #   "labels": [...],
    #   "filter": "ip",
    #   "confidence_ip": 0.91
    # }

    confidence_ip = float(model_result.get("confidence_ip", 0.0))

    # -----------------------
    # 2) Auto lookup-sender
    # -----------------------
    lookup_result = await lookup_sender({
        "email": params.get("sender_email")   # optionnel si tu le veux
    })

    sender_found = lookup_result.get("found", False)

    # -----------------------
    # 3) Calcul trust score
    # -----------------------
    trust_score = compute_trust_score(
        confidence_ip=confidence_ip,
        sender_found=sender_found
    )

    # -----------------------
    # 4) Réponse consolidée
    # -----------------------
    return {
        "status": "ok",
        "email_id": email_id,
        "model_result": model_result,
        "sender_lookup": lookup_result,
        "trust_score": trust_score
    }


run_pipeline_tool = {
    "name": "run_pipeline",
    "schema": {
        "name": "run_pipeline",
        "description": "Pipeline de traitement d'email avec ML + lookup + trust-score",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_id": {"type": "integer"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "sender_email": {"type": "string"}
            },
            "required": ["email_id", "subject", "body"]
        }
    },
    "func": run_pipeline,
}
